//! Classical subject detection for the Subject, Background and People masks when the photo carries
//! no segmentation matte of its own (no model, no download).
//!
//! It runs once per photo on a small copy of the whole uncropped source ([`EDGE`] px long edge) and
//! yields a 0..255 matte over that source, used like the mattes a file carries
//! ([`crate::masks::Mattes`]): crops, rotations, lens warps and zoomed windows sample it the same
//! way, so previews, windows and exports agree. The render then snaps its edges to the
//! full-resolution photo (see `masks::shape_alpha`).
//!
//! Cues, all at the proxy resolution:
//!
//! 1. **Background from the borders.** How far a pixel is from the frame's border strip: the
//!    smaller of the minimum barrier distance (the colour spread along the best path, which unlike
//!    a sum of steps does not grow with path length, so noise and smooth gradients stay cheap) and
//!    its colour distance to the strip's palette (a few k-means colours, so textured backgrounds of
//!    the border's colours stay background). A first pass measures from each side separately and
//!    keeps the second-smallest distance (a pixel near two or more sides is background), so a
//!    subject touching one or two edges survives; the second pass measures from the border pixels
//!    the first pass left as background, so a frame-filling subject is not its own background.
//! 2. **Focus.** When the photo has a clear depth-of-field contrast, regions with little
//!    fine detail (a defocused background) lose weight.
//! 3. **A mild centre prior.**
//! 4. **Colour models.** Foreground and background colour histograms from the cues, refined twice,
//!    each pass smoothed by a guided filter on the luminance so the mask follows the photo's edges.
//! 5. **Cleanup.** Regions the cues hardly support (the colour models split any photo in two) or
//!    much smaller than the largest are dropped, and small holes filled. A photo where nothing
//!    stands out from its borders selects nothing.

use std::sync::{Arc, Mutex, PoisonError};

use lightcraft_color::perceptual::oklab_from_2020;
use lightcraft_raster::blur::gaussian;
use lightcraft_raster::resample::{Filter, fit};
use lightcraft_raster::{Image, Plane, Rgb32f};

use crate::masks::{guided_cross, tonemap_for_select};

/// Long edge of the proxy the subject is detected on (pixels).
pub const EDGE: usize = 384;

/// Colour differences (OkLab) below this are noise or smooth gradients: free to cross.
const TAU: f32 = 0.01;
/// Scale of the distances (OkLab units) at which a region counts as clearly distinct.
const SCALE: f32 = 0.08;
/// Colours per side in the border palette.
const PALETTE: usize = 10;

/// The subject matte of `src` (linear Rec.2020, as-shot white; any size): 0..255 over the whole
/// source at about [`EDGE`] px long edge. Cached per source (the last few photos).
pub fn matte(src: &Rgb32f) -> Arc<Image<u8>> {
    static CACHE: Mutex<Vec<(u64, Arc<Image<u8>>)>> = Mutex::new(Vec::new());
    let key = fingerprint(src);
    if let Some((_, m)) = CACHE.lock().unwrap_or_else(PoisonError::into_inner).iter().find(|(k, _)| *k == key) {
        return m.clone();
    }
    let proxy = fit(src, EDGE, EDGE, Filter::Box);
    let m = Arc::new(detect(&proxy));
    let mut c = CACHE.lock().unwrap_or_else(PoisonError::into_inner);
    c.retain(|(k, _)| *k != key);
    c.insert(0, (key, m.clone()));
    c.truncate(4);
    m
}

/// A cheap identity of `src`: its size and a few thousand of its pixels.
fn fingerprint(src: &Rgb32f) -> u64 {
    use std::hash::{BuildHasher, BuildHasherDefault, DefaultHasher, Hasher};
    let mut h = BuildHasherDefault::<DefaultHasher>::default().build_hasher();
    h.write_usize(src.width);
    h.write_usize(src.height);
    let step = (src.data.len() / 4096).max(1);
    for p in src.data.iter().step_by(step) {
        for v in p {
            h.write_u32(v.to_bits());
        }
    }
    h.finish()
}

/// The subject matte of a small linear Rec.2020 image (see the module docs), same size as `img`.
pub fn detect(img: &Rgb32f) -> Image<u8> {
    detect_with(img, &mut |_, _| {})
}

/// [`detect`], showing each intermediate map to `stage` (for tuning).
fn detect_with(img: &Rgb32f, stage: &mut dyn FnMut(&str, &Plane)) -> Image<u8> {
    let (w, h) = (img.width, img.height);
    let n = w * h;
    if w < 8 || h < 8 || img.data.len() != n {
        return Image::new(w, h);
    }
    let lab = to_lab(img);
    // fine texture (fur, foliage, noise) is not what separates a subject from its surroundings
    let coarse = gaussian(&lab, (0.004 * w.max(h) as f32).max(1.0));
    let lab = gaussian(&lab, 1.0);
    let l = lab.map(|c| c[0]);

    // 1. distance from each side of the frame (pass 1), then from the border pixels that are
    //    not subject (pass 2)
    let sides: Vec<Plane> = (0..4).map(|side| distance(&coarse, &border(w, h, side))).collect();
    let mut sal = Plane::new(w, h);
    for (i, v) in sal.data.iter_mut().enumerate() {
        let mut d = [0.0f32; 4];
        for (k, s) in sides.iter().enumerate() {
            d[k] = s.data.get(i).copied().unwrap_or(0.0);
        }
        d.sort_by(f32::total_cmp);
        *v = 1.0 - (-d[1] / SCALE).exp();
    }
    stage("1-sides", &sal);
    let priors = Priors::new(&l);
    let first = priors.apply(&sal);
    let t1 = otsu(&first.data);
    let strip: Vec<usize> = (0..4).flat_map(|side| border(w, h, side)).collect();
    let seeds: Vec<usize> = strip.iter().copied().filter(|&i| first.data.get(i).is_some_and(|v| *v < t1)).collect();
    if seeds.len() * 5 >= strip.len() {
        let d = distance(&coarse, &seeds);
        sal = d.map(|d| 1.0 - (-d / SCALE).exp());
    }
    stage("2-border", &sal);
    let raw_top = percentile(&sal.data, 0.99);
    let mut sal = priors.apply(&sal);
    let top = percentile(&sal.data, 0.99).max(1e-4);
    sal.data.iter_mut().for_each(|v| *v = (*v / top).min(1.0));
    let evidence = sal.clone();
    if raw_top < 0.25 {
        // nothing stands out from the borders
        return Image::new(w, h);
    }

    stage("3-priors", &sal);
    // 4. colour models, refined
    let mut cur = sal.clone();
    for _ in 0..2 {
        let t = otsu(&cur.data);
        let post = colour_posterior(&lab, &cur, t);
        let mixed = sal.zip_map(&post, |s, p| 0.65 * s + 0.35 * p);
        cur = guided_cross(&l, &mixed, 2.0, 1e-3);
    }
    let t = otsu(&cur.data).clamp(0.2, 0.8);
    let mut alpha = cur.map(|v| smooth(t - 0.08, t + 0.08, v));

    stage("4-colour", &alpha);
    // 5. cleanup
    cleanup(&mut alpha, &evidence);
    stage("5-final", &alpha);
    alpha.map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8)
}

#[inline]
fn smooth(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// OkLab of the display-mapped image, exposed so its median luminance is mid-grey (dark and
/// bright photos compare alike).
fn to_lab(img: &Rgb32f) -> Rgb32f {
    let y = |c: &[f32; 3]| 0.2627 * c[0] + 0.678 * c[1] + 0.0593 * c[2];
    let lum: Vec<f32> = img.data.iter().map(y).filter(|v| v.is_finite()).collect();
    let med = percentile(&lum, 0.5);
    let k = if med > 1e-6 { (0.25 / med).clamp(1e-3, 1e4) } else { 1.0 };
    img.map(|c| {
        let c = c.map(|v| if v.is_finite() { (v * k).max(0.0) } else { 0.0 });
        let lab = oklab_from_2020(tonemap_for_select(c));
        if lab.iter().all(|v| v.is_finite()) { lab } else { [0.0; 3] }
    })
}

/// The pixels of a border strip (`side`: 0 top, 1 right, 2 bottom, 3 left), ~3 % of the short edge.
fn border(w: usize, h: usize, side: usize) -> Vec<usize> {
    let s = (w.min(h) / 32).max(1);
    let mut out = Vec::new();
    match side {
        0 => (0..s.min(h)).for_each(|y| out.extend((0..w).map(|x| y * w + x))),
        1 => (0..h).for_each(|y| out.extend((w.saturating_sub(s)..w).map(|x| y * w + x))),
        2 => (h.saturating_sub(s)..h).for_each(|y| out.extend((0..w).map(|x| y * w + x))),
        _ => (0..h).for_each(|y| out.extend((0..s.min(w)).map(|x| y * w + x))),
    }
    out
}

fn dist(a: [f32; 3], b: [f32; 3]) -> f32 {
    let (dl, da, db) = (a[0] - b[0], a[1] - b[1], a[2] - b[2]);
    (dl * dl + 2.0 * (da * da + db * db)).sqrt()
}

/// How far each pixel is from the `seeds` (border pixels taken as background): the smaller of
/// the minimum barrier distance and twice the colour distance to the seeds' palette.
fn distance(lab: &Rgb32f, seeds: &[usize]) -> Plane {
    let pal = palette(lab, seeds);
    barrier(lab, seeds).zip_map(lab, |g, c| g.min(2.0 * nearest(&pal, c)))
}

/// Minimum barrier distance from `seeds` (4-connected): along the best path, the spread
/// (max − min) of each OkLab channel, combined like [`dist`], less [`TAU`]. Unlike a sum of steps
/// it does not grow with path length, so noise and smooth gradients stay cheap. Raster-scan
/// approximation (three forward/backward passes).
fn barrier(lab: &Rgb32f, seeds: &[usize]) -> Plane {
    let (w, h) = (lab.width, lab.height);
    let n = w * h;
    let mut d = vec![f32::INFINITY; n];
    let mut hi: Vec<[f32; 3]> = lab.data.clone();
    let mut lo: Vec<[f32; 3]> = lab.data.clone();
    for &s in seeds {
        if let Some(v) = d.get_mut(s) {
            *v = 0.0;
        }
    }
    let spread = |u: [f32; 3], l: [f32; 3]| dist(u, l);
    let relax = |i: usize, j: usize, d: &mut [f32], hi: &mut [[f32; 3]], lo: &mut [[f32; 3]]| {
        let (Some(&dj), Some(&c)) = (d.get(j), lab.data.get(i)) else { return };
        if !dj.is_finite() {
            return;
        }
        let (Some(&uj), Some(&lj)) = (hi.get(j), lo.get(j)) else { return };
        let u = [uj[0].max(c[0]), uj[1].max(c[1]), uj[2].max(c[2])];
        let l = [lj[0].min(c[0]), lj[1].min(c[1]), lj[2].min(c[2])];
        let cand = spread(u, l);
        if let Some(di) = d.get_mut(i)
            && cand < *di
        {
            *di = cand;
            if let (Some(hu), Some(ll)) = (hi.get_mut(i), lo.get_mut(i)) {
                *hu = u;
                *ll = l;
            }
        }
    };
    for _ in 0..3 {
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                if x > 0 {
                    relax(i, i - 1, &mut d, &mut hi, &mut lo);
                }
                if y > 0 {
                    relax(i, i - w, &mut d, &mut hi, &mut lo);
                }
            }
        }
        for y in (0..h).rev() {
            for x in (0..w).rev() {
                let i = y * w + x;
                if x + 1 < w {
                    relax(i, i + 1, &mut d, &mut hi, &mut lo);
                }
                if y + 1 < h {
                    relax(i, i + w, &mut d, &mut hi, &mut lo);
                }
            }
        }
    }
    let data = d.into_iter().map(|v| if v.is_finite() { (v - TAU).max(0.0) } else { 0.0 }).collect();
    Image { width: w, height: h, data }
}

/// The focus and centre priors, applied to a saliency map.
struct Priors {
    focus: Option<(Plane, f32)>,
}

impl Priors {
    fn new(l: &Plane) -> Priors {
        Priors { focus: focus(l) }
    }

    fn apply(&self, sal: &Plane) -> Plane {
        let (w, h) = (sal.width, sal.height);
        let long = w.max(h).max(1) as f32;
        let mut out = sal.clone();
        for (i, v) in out.data.iter_mut().enumerate() {
            let (x, y) = ((i % w) as f32 + 0.5, (i / w) as f32 + 0.5);
            let (dx, dy) = ((x - w as f32 * 0.5) / long, (y - h as f32 * 0.5) / long);
            let centre = (-(dx * dx + dy * dy) / (2.0 * 0.3 * 0.3)).exp();
            let f = match &self.focus {
                Some((p, k)) => 1.0 - k * 0.7 * (1.0 - p.data.get(i).copied().unwrap_or(1.0)),
                None => 1.0,
            };
            *v *= (0.6 + 0.4 * centre) * f;
        }
        out
    }
}

/// A few representative colours of `seeds` (k-means, deterministic start).
fn palette(lab: &Rgb32f, seeds: &[usize]) -> Vec<[f32; 3]> {
    let px: Vec<[f32; 3]> = seeds.iter().filter_map(|&i| lab.data.get(i).copied()).collect();
    if px.is_empty() {
        return Vec::new();
    }
    let k = PALETTE.min(px.len());
    let mut c: Vec<[f32; 3]> = (0..k).filter_map(|j| px.get(j * px.len() / k).copied()).collect();
    for _ in 0..6 {
        let mut acc = vec![([0.0f64; 3], 0usize); c.len()];
        for p in &px {
            let j = (0..c.len()).min_by(|&a, &b| dist(*p, c[a]).total_cmp(&dist(*p, c[b]))).unwrap_or(0);
            if let Some(a) = acc.get_mut(j) {
                for (s, v) in a.0.iter_mut().zip(p) {
                    *s += *v as f64;
                }
                a.1 += 1;
            }
        }
        for (cj, (s, n)) in c.iter_mut().zip(&acc) {
            if *n > 0 {
                *cj = s.map(|v| (v / *n as f64) as f32);
            }
        }
    }
    c
}

fn nearest(pal: &[[f32; 3]], c: [f32; 3]) -> f32 {
    pal.iter().map(|p| dist(*p, c)).fold(f32::INFINITY, f32::min).min(10.0)
}

/// Fine-detail energy, pooled and normalized to 0..1, and how strongly the photo separates
/// sharp from defocused regions (0: no depth-of-field contrast). `None` on a flat image.
fn focus(l: &Plane) -> Option<(Plane, f32)> {
    let fine = l.zip_map(&gaussian(l, 1.0), |a, b| (a - b).abs());
    let long = l.width.max(l.height) as f32;
    let pooled = gaussian(&fine, 0.03 * long);
    let (lo, hi) = (percentile(&pooled.data, 0.2), percentile(&pooled.data, 0.9));
    if hi <= 1e-5 {
        return None;
    }
    let k = smooth(3.0, 8.0, hi / lo.max(1e-6));
    Some((pooled.map(|v| smooth(lo, hi, v)), k))
}

/// For each pixel, the probability that its colour belongs to the foreground, from colour
/// histograms of the pixels above / below threshold `t` of `cur` (soft weights).
fn colour_posterior(lab: &Rgb32f, cur: &Plane, t: f32) -> Plane {
    const BL: usize = 10;
    const BA: usize = 12;
    let bin = |c: [f32; 3]| {
        let q = |v: f32, lo: f32, hi: f32, n: usize| (((v - lo) / (hi - lo)).clamp(0.0, 0.9999) * n as f32) as usize;
        (q(c[0], 0.0, 1.0, BL) * BA + q(c[1], -0.25, 0.25, BA)) * BA + q(c[2], -0.25, 0.25, BA)
    };
    let mut hf = vec![0.0f32; BL * BA * BA];
    let mut hb = vec![0.0f32; BL * BA * BA];
    let (mut nf, mut nb) = (0.0f32, 0.0f32);
    for (c, s) in lab.data.iter().zip(&cur.data) {
        let b = bin(*c);
        let (wf, wb) = (smooth(t + 0.05, t + 0.3, *s), 1.0 - smooth(t - 0.2, t, *s));
        if let (Some(f), Some(g)) = (hf.get_mut(b), hb.get_mut(b)) {
            *f += wf;
            *g += wb;
        }
        nf += wf;
        nb += wb;
    }
    let (nf, nb) = (nf.max(1.0), nb.max(1.0));
    let eps = 0.5;
    lab.map(|c| {
        let b = bin(c);
        let pf = (hf.get(b).copied().unwrap_or(0.0) + eps) / nf;
        let pb = (hb.get(b).copied().unwrap_or(0.0) + eps) / nb;
        pf / (pf + pb)
    })
}

/// Otsu's threshold of values in 0..1.
fn otsu(v: &[f32]) -> f32 {
    let mut hist = [0u32; 64];
    for x in v {
        let b = (x.clamp(0.0, 0.9999) * 64.0) as usize;
        if let Some(h) = hist.get_mut(b) {
            *h += 1;
        }
    }
    let total = v.len().max(1) as f64;
    let sum: f64 = hist.iter().enumerate().map(|(i, &c)| i as f64 * c as f64).sum();
    let (mut wb, mut sb, mut best, mut at) = (0.0f64, 0.0f64, -1.0f64, 32usize);
    for (i, &c) in hist.iter().enumerate() {
        wb += c as f64;
        sb += i as f64 * c as f64;
        let wf = total - wb;
        if wb == 0.0 || wf == 0.0 {
            continue;
        }
        let (mb, mf) = (sb / wb, (sum - sb) / wf);
        let between = wb * wf * (mb - mf) * (mb - mf);
        if between > best {
            best = between;
            at = i;
        }
    }
    (at as f32 + 1.0) / 64.0
}

fn percentile(v: &[f32], q: f32) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    let mut s: Vec<f32> = v.iter().copied().filter(|x| x.is_finite()).collect();
    if s.is_empty() {
        return 0.0;
    }
    let k = ((s.len() - 1) as f32 * q.clamp(0.0, 1.0)) as usize;
    let (_, m, _) = s.select_nth_unstable_by(k, f32::total_cmp);
    *m
}

/// Drop selected regions that are much smaller than the largest or that the cues hardly support
/// (mean normalized `evidence` below 0.25: the colour models split any photo in two, a subject or not); fill
/// small unselected holes; soften the result by a pixel.
fn cleanup(alpha: &mut Plane, evidence: &Plane) {
    let (w, h) = (alpha.width, alpha.height);
    let n = w * h;
    let fg: Vec<bool> = alpha.data.iter().map(|v| *v > 0.5).collect();
    let (lab, sizes, touches) = components(&fg, w, h);
    let is_fg = fg_of(&lab, &fg, sizes.len());
    let mut support = vec![0.0f32; sizes.len()];
    for (c, e) in lab.iter().zip(&evidence.data) {
        if let Some(s) = support.get_mut(*c) {
            *s += e;
        }
    }
    let supported: Vec<bool> = support.iter().zip(&sizes).map(|(s, n)| *s / (*n).max(1) as f32 >= 0.25).collect();
    let largest = sizes.iter().zip(is_fg.iter().zip(&supported)).filter(|(_, (f, s))| **f && **s).map(|(n, _)| *n).max().unwrap_or(0);
    let keep_min = (largest as f32 * 0.15).max(n as f32 * 0.002) as usize;
    // what to drop / fill, grown by a couple of pixels so the soft rims go with them
    let mut drop = Plane::new(w, h);
    let mut fill = Plane::new(w, h);
    for (i, &c) in lab.iter().enumerate() {
        let size = sizes.get(c).copied().unwrap_or(0);
        let f = is_fg.get(c).copied().unwrap_or(false);
        let edge = touches.get(c).copied().unwrap_or(true);
        let ok = supported.get(c).copied().unwrap_or(false);
        let target = if f && (size < keep_min || !ok) {
            Some(&mut drop)
        } else if !f && !edge && size < n / 50 && largest > 0 {
            Some(&mut fill)
        } else {
            None
        };
        if let Some(d) = target.and_then(|t| t.data.get_mut(i)) {
            *d = 1.0;
        }
    }
    let (drop, fill) = (gaussian(&drop, 1.5), gaussian(&fill, 1.5));
    for ((v, d), f) in alpha.data.iter_mut().zip(&drop.data).zip(&fill.data) {
        if *f > 0.05 {
            *v = v.max(smooth(0.05, 0.3, *f));
        }
        if *d > 0.05 {
            *v *= 1.0 - smooth(0.05, 0.3, *d);
        }
    }
    *alpha = gaussian(alpha, 0.7);
}

/// Whether each component is foreground.
fn fg_of(lab: &[usize], fg: &[bool], count: usize) -> Vec<bool> {
    let mut out = vec![false; count];
    for (c, f) in lab.iter().zip(fg) {
        if let Some(o) = out.get_mut(*c) {
            *o = *f;
        }
    }
    out
}

/// 4-connected components of equal `fg` value: per-pixel label, sizes, whether each touches the
/// frame's edge.
fn components(fg: &[bool], w: usize, h: usize) -> (Vec<usize>, Vec<usize>, Vec<bool>) {
    let n = w * h;
    let mut lab = vec![usize::MAX; n];
    let (mut sizes, mut touches) = (Vec::new(), Vec::new());
    let mut stack = Vec::new();
    for start in 0..n {
        if lab.get(start).copied() != Some(usize::MAX) {
            continue;
        }
        let id = sizes.len();
        let val = fg.get(start).copied().unwrap_or(false);
        let (mut size, mut edge) = (0usize, false);
        stack.push(start);
        if let Some(l) = lab.get_mut(start) {
            *l = id;
        }
        while let Some(i) = stack.pop() {
            size += 1;
            let (x, y) = (i % w, i / w);
            edge |= x == 0 || y == 0 || x + 1 == w || y + 1 == h;
            let nb = [(x > 0).then(|| i - 1), (x + 1 < w).then(|| i + 1), (y > 0).then(|| i - w), (y + 1 < h).then(|| i + w)];
            for j in nb.into_iter().flatten() {
                if fg.get(j).copied() == Some(val)
                    && let Some(l) = lab.get_mut(j)
                    && *l == usize::MAX
                {
                    *l = id;
                    stack.push(j);
                }
            }
        }
        sizes.push(size);
        touches.push(edge);
    }
    (lab, sizes, touches)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A textured backdrop (two colours, a gradient) with a red disc off-centre.
    fn scene(w: usize, h: usize, cx: f32, cy: f32, r: f32) -> Rgb32f {
        Rgb32f::from_fn(w, h, |x, y| {
            let (fx, fy) = (x as f32 / w as f32, y as f32 / h as f32);
            let d = ((fx - cx) * w as f32).hypot((fy - cy) * h as f32) / w.max(h) as f32;
            if d < r {
                [0.5, 0.05, 0.04]
            } else if fy < 0.55 {
                [0.12 + 0.1 * fy, 0.2 + 0.1 * fy, 0.45]
            } else {
                [0.08, 0.16 + 0.02 * ((x / 3 + y / 3) % 2) as f32, 0.05]
            }
        })
    }

    fn at(m: &Image<u8>, fx: f32, fy: f32) -> u8 {
        m.get((fx * m.width as f32) as usize, (fy * m.height as f32) as usize)
    }

    /// Tuning aid: `LC_SUBJECT_IN=photo.jpg cargo test -p lightcraft-pipeline dump_stages -- --ignored`
    /// writes each stage next to the input.
    #[test]
    #[ignore]
    fn dump_stages() {
        let Ok(path) = std::env::var("LC_SUBJECT_IN") else { return };
        let rgb = image::open(&path).unwrap().to_rgb32f();
        let lin = |v: f32| if v <= 0.04045 { v / 12.92 } else { ((v + 0.055) / 1.055).powf(2.4) };
        let src = Rgb32f::from_fn(rgb.width() as usize, rgb.height() as usize, |x, y| rgb.get_pixel(x as u32, y as u32).0.map(lin));
        let proxy = fit(&src, EDGE, EDGE, Filter::Box);
        let save = |name: &str, p: &Plane| {
            let top = p.data.iter().copied().fold(1e-6f32, f32::max);
            let g =
                image::GrayImage::from_fn(p.width as u32, p.height as u32, |x, y| image::Luma([(p.get(x as usize, y as usize) / top * 255.0) as u8]));
            g.save(format!("{path}.{name}.png")).unwrap();
        };
        detect_with(&proxy, &mut |n, p| save(n, p));
    }

    #[test]
    fn finds_a_distinct_object() {
        let m = detect(&scene(240, 160, 0.42, 0.5, 0.14));
        assert!(at(&m, 0.42, 0.5) > 200, "centre {}", at(&m, 0.42, 0.5));
        for (x, y) in [(0.05, 0.1), (0.9, 0.2), (0.9, 0.9), (0.1, 0.9), (0.75, 0.5)] {
            assert!(at(&m, x, y) < 40, "background at {x},{y}: {}", at(&m, x, y));
        }
    }

    #[test]
    fn a_subject_touching_one_edge_is_kept() {
        // a tall red post rising from the bottom edge
        let img = Rgb32f::from_fn(200, 200, |x, y| if (90..110).contains(&x) && y > 60 { [0.5, 0.05, 0.04] } else { [0.15, 0.25, 0.45] });
        let m = detect(&img);
        assert!(at(&m, 0.5, 0.7) > 200, "{}", at(&m, 0.5, 0.7));
        assert!(at(&m, 0.2, 0.7) < 40);
    }

    #[test]
    fn flat_or_tiny_inputs_select_nothing_and_never_panic() {
        assert!(detect(&Rgb32f::filled(64, 48, [0.2, 0.2, 0.2])).data.iter().all(|v| *v == 0));
        assert!(detect(&Rgb32f::new(3, 2)).data.iter().all(|v| *v == 0));
        assert_eq!(detect(&Rgb32f::new(0, 0)).data.len(), 0);
        let hostile = Rgb32f::from_fn(40, 30, |x, y| match (x + y) % 3 {
            0 => [f32::NAN, 1.0, 0.0],
            1 => [f32::INFINITY, -5.0, 1e30],
            _ => [0.1, 0.2, 0.3],
        });
        let _ = detect(&hostile);
    }

    /// A Background mask darkens around the detected subject, at any output size (the estimate
    /// is made once over the whole source, not per render).
    #[test]
    fn a_background_mask_darkens_around_the_subject_at_any_size() {
        use lightcraft_develop::{DevelopSettings, LocalAdjustments, Mask, MaskComponent, MaskOp, MaskShape};
        let src = scene(480, 320, 0.42, 0.5, 0.14);
        let mut s = DevelopSettings::default();
        let adjust = LocalAdjustments { exposure: -3.0, ..Default::default() };
        let comp = MaskComponent { name: None, op: MaskOp::Add, invert: false, shape: MaskShape::Background };
        s.masks.push(Mask { id: 1, components: vec![comp], adjust, ..Default::default() });
        let luma = |p: [u8; 4]| 0.2126 * p[0] as f32 + 0.7152 * p[1] as f32 + 0.0722 * p[2] as f32;
        for (w, h) in [(240, 160), (480, 320)] {
            let req = crate::RenderRequest::fit(w, h);
            let plain = crate::render(&src, &crate::SourceInfo::default(), &DevelopSettings::default(), &req).image;
            let masked = crate::render(&src, &crate::SourceInfo::default(), &s, &req).image;
            let ratio = |fx: f32, fy: f32| {
                let i = (fy * h as f32) as usize * w + (fx * w as f32) as usize;
                luma(masked.data[i]) / luma(plain.data[i]).max(1.0)
            };
            assert!(ratio(0.42, 0.5) > 0.9, "subject darkened at {w}: {}", ratio(0.42, 0.5));
            for (x, y) in [(0.1, 0.15), (0.85, 0.3), (0.85, 0.85)] {
                assert!(ratio(x, y) < 0.6, "background kept at {w} ({x},{y}): {}", ratio(x, y));
            }
        }
    }

    #[test]
    fn the_matte_does_not_depend_on_the_source_resolution() {
        let (a, b) = (matte(&scene(1536, 1024, 0.6, 0.45, 0.12)), matte(&scene(768, 512, 0.6, 0.45, 0.12)));
        assert_eq!((a.width, a.height), (b.width, b.height));
        let diff = a.data.iter().zip(&b.data).filter(|(x, y)| x.abs_diff(**y) > 64).count();
        assert!(diff * 100 < a.data.len(), "{diff} of {} differ", a.data.len());
    }
}
