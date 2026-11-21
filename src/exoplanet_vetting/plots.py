"""Figures drawn for the repository, not copied from the article."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def save_phase_fold_gif(path: str | Path, frames: int = 48) -> None:
    """Animate a light curve folding onto its transit.

    The motion follows the phase-folding sketch in the Astronet repository.
    The samples and the rendering are original.
    """

    import matplotlib.pyplot as plt
    from matplotlib.animation import PillowWriter

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    orbits = 4
    samples = 720
    time = np.linspace(0.0, orbits, samples, endpoint=False)
    # Mid-transit at phase 0.5, so the dip sits in the middle once the orbits stack.
    phase = np.mod(time, 1.0)
    distance = np.abs(phase - 0.5)
    transit = distance < 0.04
    flux = 1.0 + 0.004 * np.sin(2 * np.pi * time * 1.7)
    flux -= 0.018 * np.exp(-0.5 * (distance / 0.012) ** 2)
    flux += rng.normal(0.0, 0.0015, size=samples)

    figure, axes = plt.subplots(figsize=(7.4, 3.5), dpi=120)
    figure.patch.set_facecolor("#07080b")
    axes.set_facecolor("#07080b")
    for spine in axes.spines.values():
        spine.set_color("#2a3140")
    axes.tick_params(colors="#9aa3b2", labelsize=9)
    axes.set_ylabel("Flux", color="#c5ccd6")
    axes.set_ylim(0.965, 1.012)
    scatter = axes.scatter([], [], s=8, linewidths=0)
    caption = axes.text(0.0, 1.02, "", transform=axes.transAxes, color="#d5dbe3", fontsize=12)

    def draw(frame: int):
        progress = frame / (frames - 1)
        # Hold the time series, fold, then hold the folded curve.
        if progress < 0.2:
            mix = 0.0
            label = "Unfolded light curve"
            xlabel = "Time (orbits)"
            left, right = -0.05, orbits + 0.05
        elif progress < 0.8:
            mix = (progress - 0.2) / 0.6
            mix = mix * mix * (3 - 2 * mix)
            label = "Phase folding"
            xlabel = "Time  →  phase"
            left = -0.05 * (1 - mix) + (-0.02) * mix
            right = (orbits + 0.05) * (1 - mix) + 1.02 * mix
        else:
            mix = 1.0
            label = "Phase-folded transit"
            xlabel = "Phase"
            left, right = -0.02, 1.02
        position = (1 - mix) * time + mix * phase
        color = np.where(transit, "#e07a3d", "#e8edf2")
        order = np.argsort(position)
        scatter.set_offsets(np.column_stack((position[order], flux[order])))
        scatter.set_color(color[order])
        axes.set_xlim(left, right)
        axes.set_xlabel(xlabel, color="#c5ccd6")
        caption.set_text(label)
        return (scatter, caption)

    writer = PillowWriter(fps=12)
    with writer.saving(figure, str(path), dpi=120):
        for frame in range(frames):
            draw(frame)
            writer.grab_frame()
    plt.close(figure)


def _font(size: int) -> ImageFont.ImageFont:
    for name in ("segoeuil.ttf", "segoeui.ttf", "arial.ttf"):
        path = Path(r"C:\Windows\Fonts") / name
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def _ease(value: float) -> float:
    value = min(1.0, max(0.0, value))
    return value * value * (3.0 - 2.0 * value)


def _value_noise(height: int, width: int, cells_y: int, cells_x: int, rng: np.random.Generator) -> np.ndarray:
    # Longitude wraps, so the left and right edges of the map are the same line on the star.
    grid = rng.random((cells_y + 2, cells_x)).astype(np.float32)
    grid = np.concatenate([grid, grid[:, :1]], axis=1)
    ys = np.linspace(0, cells_y, height, endpoint=False)
    xs = np.linspace(0, cells_x, width, endpoint=False)
    y0 = ys.astype(np.int32)
    x0 = np.floor(xs).astype(np.int32) % cells_x
    x1 = (x0 + 1) % cells_x
    fy = (ys - y0)[:, None]
    fx = (xs - np.floor(xs))[None, :]
    return (
        grid[y0[:, None], x0[None, :]] * (1 - fy) * (1 - fx)
        + grid[y0[:, None], x1[None, :]] * (1 - fy) * fx
        + grid[y0[:, None] + 1, x0[None, :]] * fy * (1 - fx)
        + grid[y0[:, None] + 1, x1[None, :]] * fy * fx
    )


def _photosphere(rng: np.random.Generator) -> np.ndarray:
    """Turbulent cyan photosphere, in the manner of a hot luminous star."""
    height, width = 240, 560
    field = np.zeros((height, width), dtype=np.float32)
    weight = 0.0
    for cells, amplitude in ((28, 0.45), (60, 0.85), (120, 1.0), (200, 0.7)):
        field += amplitude * _value_noise(height, width, max(8, cells // 3), cells, rng)
        weight += amplitude
    field /= weight
    grain = np.clip((field - 0.40) * 3.4, 0, 1)
    cyan = np.array([0.00, 0.62, 1.00], dtype=np.float32)
    bright = np.array([0.35, 0.88, 1.00], dtype=np.float32)
    white = np.array([0.92, 0.98, 1.00], dtype=np.float32)
    color = cyan + (bright - cyan) * grain[..., None]
    color += (white - color) * (grain**4)[..., None] * 0.9
    return np.clip(color, 0, 1)


def _sample(texture: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    height, width = texture.shape[:2]
    x = np.mod(u, 1.0) * width
    y = np.clip(v, 0.0, 1.0) * (height - 1)
    x0 = np.floor(x).astype(np.int32) % width
    y0 = np.clip(np.floor(y).astype(np.int32), 0, height - 1)
    x1 = (x0 + 1) % width
    y1 = np.clip(y0 + 1, 0, height - 1)
    fx = (x - np.floor(x))[..., None]
    fy = (y - np.floor(y))[..., None]
    return (
        texture[y0, x0] * (1 - fx) * (1 - fy)
        + texture[y0, x1] * fx * (1 - fy)
        + texture[y1, x0] * (1 - fx) * fy
        + texture[y1, x1] * fx * fy
    )


def _star_sprite(
    diameter: int,
    angle: float,
    morph: float,
    textures: tuple[np.ndarray, np.ndarray],
    turn: float,
) -> np.ndarray:
    center = (diameter - 1) / 2
    yy, xx = np.indices((diameter, diameter), dtype=np.float32)
    nx = (xx - center) / (diameter / 2)
    ny = (center - yy) / (diameter / 2)
    radius2 = nx**2 + ny**2
    facing = np.sqrt(np.clip(1 - radius2, 0, 1))
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    xr = nx * cos_a + facing * sin_a
    zr = -nx * sin_a + facing * cos_a
    longitude = (np.arctan2(xr, zr) + np.pi) / (2 * np.pi)
    latitude = (np.arcsin(np.clip(ny, -1, 1)) + np.pi / 2) / np.pi
    color = _sample(textures[0], longitude, latitude) * (1 - morph) + _sample(textures[1], longitude, latitude) * morph
    # Spots are fixed on the rotating surface, so they travel with the granules.
    for lon0, lat0, radius, phase in ((0.22, 0.42, 0.04, 0.5), (0.58, 0.60, 0.048, 2.1), (0.84, 0.34, 0.032, 4.0)):
        grow = 0.72 + 0.28 * np.sin(2 * np.pi * turn + phase)
        dlon = (longitude - lon0 + 0.5) % 1.0 - 0.5
        dlat = latitude - lat0
        metric = (dlon * np.cos((latitude - 0.5) * np.pi)) ** 2 + dlat**2
        color *= 1 - 0.5 * np.exp(-metric / (2 * (radius * grow) ** 2))[..., None]
    limb = 0.58 + 0.42 * facing**0.5
    color *= limb[..., None]
    color += (np.array([0.85, 0.97, 1.0]) - color) * (facing**2.4)[..., None] * 0.22
    color += (np.array([0.05, 0.45, 1.0]) - color) * ((1 - facing) * (radius2 <= 1))[..., None] * 0.55
    edge = np.clip((1 - np.sqrt(np.clip(radius2, 0, 1))) * diameter / 2, 0, 1)
    sprite = np.zeros((diameter, diameter, 4), dtype=np.float32)
    sprite[..., :3] = np.clip(color, 0, 1)
    sprite[..., 3] = (radius2 <= 1) * edge
    return sprite


def _planet_sprite(diameter: int, light: np.ndarray) -> np.ndarray:
    scale = 3
    size = (diameter + 14) * scale
    center = (size - 1) / 2
    radius = diameter * scale / 2
    yy, xx = np.indices((size, size), dtype=np.float32)
    nx = (xx - center) / radius
    ny = (center - yy) / radius
    rr = np.sqrt(nx**2 + ny**2)
    inside = rr <= 1
    nz = np.sqrt(np.clip(1 - np.minimum(rr, 1) ** 2, 0, None))
    normal = np.stack((nx, ny, nz), axis=-1)
    shade = np.clip(np.sum(normal * light, axis=-1), 0, 1)
    night = np.array([0.012, 0.018, 0.04], dtype=np.float32)
    day = np.array([0.84, 0.91, 1.00], dtype=np.float32)
    color = night + (day - night) * shade[..., None] ** 0.45
    color = np.clip(color + np.array([0.75, 0.88, 1.0]) * (shade**4)[..., None] * 0.35, 0, 1)
    body = np.clip((1.15 - rr) * 2.2, 0, 1) * inside
    halo = np.exp(-np.clip(rr - 1, 0, None) * 3.2) * (rr > 1) * (shade**1.3)
    sprite = np.zeros((size, size, 4), dtype=np.float32)
    sprite[..., :3] = np.where(inside[..., None], color, np.array([0.65, 0.82, 1.0]))
    sprite[..., 3] = np.clip(body + halo * 0.55, 0, 1)
    image = Image.fromarray((sprite * 255).astype(np.uint8), "RGBA")
    image = image.resize((diameter + 14, diameter + 14), Image.Resampling.LANCZOS)
    return np.asarray(image).astype(np.float32) / 255


def _paste(canvas: np.ndarray, sprite: np.ndarray, left: int, top: int) -> None:
    height, width = canvas.shape[:2]
    sprite_h, sprite_w = sprite.shape[:2]
    x0, y0 = max(left, 0), max(top, 0)
    x1, y1 = min(left + sprite_w, width), min(top + sprite_h, height)
    if x0 >= x1 or y0 >= y1:
        return
    crop = sprite[y0 - top : y1 - top, x0 - left : x1 - left]
    alpha = crop[..., 3:4]
    canvas[y0:y1, x0:x1] = canvas[y0:y1, x0:x1] * (1 - alpha) + crop[..., :3] * alpha


def _composite_planet(
    canvas: np.ndarray,
    sprite: np.ndarray,
    left: float,
    top: float,
    star_center: np.ndarray,
    star_radius: float,
    in_front: bool,
) -> None:
    height, width = canvas.shape[:2]
    sprite_h, sprite_w = sprite.shape[:2]
    x0, y0 = max(int(np.floor(left)), 0), max(int(np.floor(top)), 0)
    x1 = min(int(np.floor(left)) + sprite_w, width)
    y1 = min(int(np.floor(top)) + sprite_h, height)
    if x0 >= x1 or y0 >= y1:
        return
    crop = sprite[y0 - int(np.floor(top)) : y1 - int(np.floor(top)), x0 - int(np.floor(left)) : x1 - int(np.floor(left))].copy()
    ys = np.arange(y0, y1)[:, None]
    xs = np.arange(x0, x1)[None, :]
    distance = np.sqrt((xs - star_center[0]) ** 2 + (ys - star_center[1]) ** 2)
    if in_front:
        cover = np.clip((star_radius - distance) / 2.0, 0, 1)
        body = crop[..., 3] > 0.45
        darkened = crop[..., :3] * (1 - cover[..., None]) + 0.01 * cover[..., None]
        crop[..., :3] = np.where(body[..., None], darkened, crop[..., :3])
    else:
        visible = np.clip((distance - star_radius) / 2.0, 0, 1)
        crop[..., 3] *= visible
    alpha = crop[..., 3:4]
    canvas[y0:y1, x0:x1] = canvas[y0:y1, x0:x1] * (1 - alpha) + crop[..., :3] * alpha


def _text_image(text: str, font: ImageFont.ImageFont, fill: tuple[int, int, int, int], tracking: int) -> Image.Image:
    if not text:
        return Image.new("RGBA", (1, 1), (0, 0, 0, 0))
    measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    width = int(sum(measure.textlength(char, font=font) + tracking for char in text) - tracking) + 2
    image = Image.new("RGBA", (max(width, 1), font.size + 8), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    x = 0
    for char in text:
        draw.text((x, 0), char, font=font, fill=fill)
        x += measure.textlength(char, font=font) + tracking
    return image


def _fade(image: Image.Image, alpha: float) -> Image.Image:
    if alpha >= 0.999:
        return image
    red, green, blue, band = image.split()
    return Image.merge("RGBA", (red, green, blue, band.point(lambda value: int(value * alpha))))


def _covered_fraction(distance: np.ndarray, star: float, planet: float) -> np.ndarray:
    """Fraction of the stellar disk hidden by the planet."""
    distance = np.asarray(distance, dtype=np.float64)
    covered = np.zeros(distance.shape)
    contained = distance <= abs(star - planet)
    covered[contained] = np.pi * min(star, planet) ** 2
    partial = (distance > abs(star - planet)) & (distance < star + planet)
    if np.any(partial):
        separation = distance[partial]
        star2 = star * star
        planet2 = planet * planet
        star_angle = np.arccos(np.clip((separation**2 + star2 - planet2) / (2 * separation * star), -1, 1))
        planet_angle = np.arccos(np.clip((separation**2 + planet2 - star2) / (2 * separation * planet), -1, 1))
        chord = np.sqrt(np.clip((-separation + planet + star) * (separation + planet - star) * (separation - planet + star) * (separation + planet + star), 0, None))
        covered[partial] = planet2 * planet_angle + star2 * star_angle - 0.5 * chord
    return covered / (np.pi * star * star)


def save_transit_gif(path: str | Path) -> None:
    """One seamless orbit: the planet circles, crosses the star, and comes back.

    The star is a hot blue-white star. Its granules rotate, morph, and carry
    spots that grow and shrink. The brightness panel opens and closes with
    the orbit. The curve stays flat except while the planet is in front of
    the disk, and that dip is drawn deeper than a real transit so it can be seen.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1040, 520
    frames = 96
    rng = np.random.default_rng(11)
    textures = (_photosphere(rng), _photosphere(rng))
    star_px = 210
    planet_px = 40
    orbit_a = 196.0
    beta = 0.32
    star_center = np.array([268.0, 262.0])
    star_radius = star_px / 2
    panel = (600, 92, 990, 418)
    left, top, right, bottom = panel
    font = _font(15)
    small = _font(13)
    phi = np.linspace(-np.pi, np.pi, frames, endpoint=False)
    orbit_x = orbit_a * np.sin(phi)
    orbit_y = -orbit_a * np.sin(beta) * np.cos(phi)
    orbit_z = orbit_a * np.cos(beta) * np.cos(phi)
    curve_phi = np.linspace(-np.pi, np.pi, 400)
    sample_x = orbit_a * np.sin(curve_phi)
    sample_y = -orbit_a * np.sin(beta) * np.cos(curve_phi)
    sample_z = orbit_a * np.cos(beta) * np.cos(curve_phi)
    covered = _covered_fraction(np.hypot(sample_x, sample_y), star_radius, planet_px / 2)
    covered = np.where(sample_z > 0, covered, 0.0)
    full = (planet_px / 2 / star_radius) ** 2
    curve_flux = 1 - 0.048 * covered / full
    low, high = 0.935, 1.02
    measured = rng.normal(0, 0.0065, curve_phi.size)
    sky = np.zeros((height, width, 3), dtype=np.float32)
    for x, y, value in zip(rng.integers(0, width, 420), rng.integers(0, height, 420), rng.random(420) ** 2):
        sky[y, x] = (0.75, 0.84, 1.0) if rng.random() < 0.3 else (1.0, 0.96, 0.9)
        sky[y, x] *= 0.25 + 0.75 * value
    glow_y, glow_x = np.indices((height, width))
    glow_r = np.sqrt((glow_x - star_center[0]) ** 2 + (glow_y - star_center[1]) ** 2) / star_radius
    corona = np.exp(-np.clip(glow_r - 1, 0, None) * 10.0) * (glow_r < 1.35)
    sky += np.array([0.12, 0.62, 1.0]) * corona[..., None] * 0.5
    sky = np.clip(sky, 0, 1)

    def panel_open(turn: float) -> float:
        if turn < 0.15:
            return _ease(turn / 0.15)
        if turn > 0.87:
            return 1 - _ease((turn - 0.87) / 0.13)
        return 1.0

    def to_plot(sample_phi: np.ndarray, sample_flux: np.ndarray) -> np.ndarray:
        x = left + 24 + (right - left - 48) * (sample_phi + np.pi) / (2 * np.pi)
        y = top + 20 + (bottom - top - 78) * (1 - (sample_flux - low) / (high - low))
        return np.column_stack((x, y))

    curve_xy = to_plot(curve_phi, curve_flux)
    baseline_y = float(to_plot(np.array([0.0]), np.array([1.0]))[0, 1])
    images: list[Image.Image] = []
    for frame in range(frames):
        angle = 2 * np.pi * frame / frames
        morph = 0.22 * (1 - np.cos(angle))
        canvas = sky.copy()
        star = _star_sprite(star_px, angle, morph, textures, frame / frames)
        _paste(canvas, star, int(star_center[0] - star_px / 2), int(star_center[1] - star_px / 2))
        light = -np.array([orbit_x[frame], orbit_y[frame], orbit_z[frame]], dtype=np.float32)
        light /= max(float(np.linalg.norm(light)), 1e-6)
        sprite = _planet_sprite(planet_px, light)
        _composite_planet(
            canvas,
            sprite,
            star_center[0] + orbit_x[frame] - sprite.shape[1] / 2,
            star_center[1] + orbit_y[frame] - sprite.shape[0] / 2,
            star_center,
            star_radius - 1,
            bool(orbit_z[frame] > 0),
        )
        image = Image.fromarray((np.clip(canvas, 0, 1) * 255).astype(np.uint8), "RGB").convert("RGBA")
        draw = ImageDraw.Draw(image)
        opened = panel_open(frame / frames)
        cx = (left + right) / 2
        cy = (top + bottom) / 2
        grow_x = _ease(min(1.0, opened / 0.42))
        grow_y = _ease(min(1.0, max(0.0, (opened - 0.22) / 0.78)))
        box = (
            cx - (right - left) / 2 * grow_x,
            cy - (bottom - top) / 2 * grow_y,
            cx + (right - left) / 2 * grow_x,
            cy + (bottom - top) / 2 * grow_y,
        )
        if grow_x > 0.02:
            draw.rectangle(box, outline=(214, 228, 242, 230), width=2)
            tick = 14 * grow_y
            for corner_x, corner_y, sign_x, sign_y in (
                (box[0], box[1], 1, 1),
                (box[2], box[1], -1, 1),
                (box[0], box[3], 1, -1),
                (box[2], box[3], -1, -1),
            ):
                draw.line((corner_x, corner_y, corner_x + sign_x * tick, corner_y), fill=(150, 196, 232, 255), width=1)
                draw.line((corner_x, corner_y, corner_x, corner_y + sign_y * tick), fill=(150, 196, 232, 255), width=1)
        if opened > 0.96:
            draw.line((left, baseline_y, right, baseline_y), fill=(150, 168, 188, 210), width=1)
            mid_x = float(curve_xy[len(curve_xy) // 2, 0])
            y_dot = top + 14
            while y_dot < bottom - 56:
                draw.line((mid_x, y_dot, mid_x, y_dot + 3), fill=(186, 204, 220, 200), width=1)
                y_dot += 9
            shown = int(np.searchsorted(curve_phi, phi[frame], side="right"))
            if shown > 1:
                pts = [tuple(point) for point in curve_xy[:shown]]
                draw.line(pts, fill=(232, 242, 255, 255), width=3, joint="curve")
                head = pts[-1]
                draw.ellipse((head[0] - 3.5, head[1] - 3.5, head[0] + 3.5, head[1] + 3.5), fill=(255, 255, 255, 255))
            reveal = min(1.0, max(0.0, (frame / frames - 0.56) / 0.1))
            visible = int(curve_phi.size * reveal * min(1.0, shown / max(curve_phi.size * 0.5, 1)))
            for index in range(visible):
                x, y = curve_xy[index]
                y += measured[index] / (high - low) * (bottom - top - 78)
                draw.ellipse((x - 2.6, y - 2.6, x + 2.6, y + 2.6), fill=(78, 198, 222, 235))
            if reveal > 0.55:
                label = _fade(_text_image("RAW DATA", small, (78, 198, 222, 255), 3), min(1.0, (reveal - 0.55) / 0.3))
                image.alpha_composite(label, (int((left + right - label.size[0]) / 2), bottom - 34))
        text_progress = _ease(min(1.0, max(0.0, (opened - 0.62) / 0.38)))
        if text_progress > 0.04:
            count = max(1, int(round(text_progress * len("BRIGHTNESS"))))
            brightness = _text_image("BRIGHTNESS"[:count], font, (214, 226, 238, 255), 2)
            brightness = _fade(brightness, min(1.0, text_progress * 1.4)).rotate(90, expand=True)
            image.alpha_composite(brightness, (left - brightness.size[0] - 8, int((top + bottom - brightness.size[1]) / 2)))
            count = max(1, int(round(text_progress * len("TIME IN HOURS"))))
            hours = _fade(_text_image("TIME IN HOURS"[:count], font, (214, 226, 238, 255), 2), min(1.0, text_progress * 1.4))
            image.alpha_composite(hours, (int((left + right - hours.size[0]) / 2), bottom + 12))
        images.append(image.convert("RGB"))

    master = images[int(frames * 0.68)].quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    quantized = [frame.quantize(palette=master, dither=Image.Dither.NONE) for frame in images]
    quantized[0].save(
        path,
        save_all=True,
        append_images=quantized[1:],
        duration=70,
        loop=0,
        optimize=True,
        disposal=2,
    )
