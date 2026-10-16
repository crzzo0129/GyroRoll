"""Render side/front/oblique views of the upper-flywheel layout."""

from pathlib import Path
import json
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter


ROOT = Path(__file__).resolve().parent


def main():
    # Studio lighting is presentation-only; the saved simulation stays untouched.
    xml = ET.parse(ROOT / "nested_model.xml").getroot()
    asset = xml.find("asset")
    ET.SubElement(asset, "texture", name="studio_sky", type="skybox", builtin="gradient",
                  rgb1=".88 .91 .92", rgb2=".96 .97 .97", width="256", height="1536")
    ET.SubElement(asset, "material", name="studio_floor", rgba=".87 .90 .91 1",
                  reflectance="0", specular=".1")
    xml.find("worldbody/geom[@name='ground']").set("material", "studio_floor")
    model = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding="unicode"))
    model.vis.global_.offwidth = 720
    model.vis.global_.offheight = 800
    model.vis.quality.shadowsize = 4096
    # Avoid multisample blending of integer segmentation IDs on Windows GL.
    model.vis.quality.offsamples = 0
    model.vis.headlight.ambient[:] = (.35, .35, .35)
    model.vis.headlight.diffuse[:] = (.6, .6, .6)
    data = mujoco.MjData(model)
    config = json.loads((ROOT / "nested_controller.json").read_text())
    mujoco.mj_forward(model, data)
    canvas = Image.new("RGB", (2160, 1080), "#eef2f3")
    draw = ImageDraw.Draw(canvas)
    def font(size):
        for path in (Path("C:/Windows/Fonts/bahnschrift.ttf"),
                     Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")):
            if path.exists():
                return ImageFont.truetype(str(path), size)
        return ImageFont.load_default(size=size)
    draw.text((48, 24), "GYROROLL / CARGO", font=font(48), fill="#152a31")
    draw.text((50, 85), "COVER REMOVED  /  SINGLE ROTOR MARK  /  24 mm WIDE TIRE", font=font(21), fill="#52656b")
    with mujoco.Renderer(model, height=800, width=720) as renderer:
        for index, (azimuth, label) in enumerate(((145, "01  /  PERSPECTIVE"),
                                                 (180, "02  /  FRONT"),
                                                 (90, "03  /  SIDE"))):
            camera = mujoco.MjvCamera()
            camera.lookat[:] = (0., 0., .30)
            camera.distance = 1.05
            camera.azimuth = azimuth
            camera.elevation = -28 if index == 0 else 0
            renderer.update_scene(data, camera=camera)
            pixels = renderer.render().copy()
            renderer.enable_segmentation_rendering()
            segmentation = renderer.render().copy()
            renderer.disable_segmentation_rendering()
            mask = ((segmentation[:, :, 1] == mujoco.mjtObj.mjOBJ_GEOM)
                    & (segmentation[:, :, 0] != model.geom("ground").id)
                    & (segmentation[:, :, 0] >= 0))
            # Composite the actual model over a clean studio sweep. A soft
            # presentation shadow replaces the hard tessellated floor shadow.
            gradient = np.linspace([238, 242, 243], [225, 232, 234], 800).astype(np.uint8)
            panel = Image.fromarray(np.broadcast_to(gradient[:, None, :], (800, 720, 3)).copy())
            ys, xs = np.nonzero(mask)
            floor_y = int(ys.max())
            center_x = int(np.mean(xs[ys > floor_y - 12]))
            shadow = Image.new("RGBA", panel.size)
            ImageDraw.Draw(shadow).ellipse((center_x - 85, floor_y - 9, center_x + 85, floor_y + 12),
                                           fill=(31, 49, 55, 55))
            panel = Image.alpha_composite(panel.convert("RGBA"), shadow.filter(ImageFilter.GaussianBlur(12)))
            panel.paste(Image.fromarray(pixels), (0, 0), Image.fromarray(mask.astype(np.uint8) * 255))
            canvas.paste(panel.convert("RGB"), (720 * index, 140))
            draw.text((720 * index + 48, 950), label, font=font(22), fill="#152a31")
    draw.line((48, 996, 2112, 996), fill="#b6c8cb", width=2)
    draw.text((48, 1016), f"DRIVE  {2000 * config['outer_radius']:g} mm     /     "
              f"ROTOR  {2000 * config['flywheel_radius']:g} mm     /     "
              "EDGE GAP  50 mm     /     HEIGHT  580 mm     /     TRAY  156 x 216 mm", font=font(23), fill="#52656b")
    canvas.save(ROOT / "upper_flywheel_preview.png")
    canvas.save(ROOT / "appearance_preview.png")
    canvas.save(ROOT / "structure_preview.png")


if __name__ == "__main__":
    main()
