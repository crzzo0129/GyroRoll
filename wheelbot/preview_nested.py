"""Render side/front/oblique views of the nested layout without a viewer."""

from pathlib import Path
import json
import mujoco
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent


def main():
    model = mujoco.MjModel.from_xml_path(str(ROOT / "nested_model.xml"))
    model.vis.global_.offwidth = 720
    model.vis.global_.offheight = 640
    data = mujoco.MjData(model)
    config = json.loads((ROOT / "nested_controller.json").read_text())
    mujoco.mj_forward(model, data)
    canvas = Image.new("RGB", (2160, 710), "#f5f7f8")
    draw = ImageDraw.Draw(canvas)
    with mujoco.Renderer(model, height=640, width=720) as renderer:
        for index, (azimuth, label) in enumerate(((90, "SIDE / drive axis Y"),
                                                 (0, "FRONT / flywheel axis X"),
                                                 (130, "OBLIQUE / exposed lower flywheel"))):
            camera = mujoco.MjvCamera()
            camera.lookat[:] = (0., 0., .27)
            camera.distance = 1.02
            camera.azimuth = azimuth
            camera.elevation = -8
            renderer.update_scene(data, camera=camera)
            canvas.paste(Image.fromarray(renderer.render()), (720 * index, 40))
            draw.text((720 * index + 24, 14), label, fill="#1b393f")
    draw.text((24, 687), f"Outer wheel: {2000 * config['outer_radius']:g} mm | "
              f"Clear bore: {2000 * config['inner_radius']:g} mm | "
              f"Flywheel: {2000 * config['flywheel_radius']:g} mm | "
              f"Rim width: {2000 * config['rim_halfwidth']:g} mm | Perpendicular axes", fill="#1b393f")
    canvas.save(ROOT / "nested_preview.png")


if __name__ == "__main__":
    main()
