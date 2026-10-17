"""Massless cosmetic surfaces for the upper-flywheel robot."""

import xml.etree.ElementTree as ET


def cosmetic(body, name, **attributes):
    return ET.SubElement(body, "geom", name=f"visual_{name}", mass="0",
                         contype="0", conaffinity="0", group="1", **attributes)


def set_cover_visible(root, visible=False):
    """Display the +X cover without changing its mass or collision geometry."""
    base = root.find("worldbody/body[@name='base']")
    prefixes = ("visual_shell_1", "visual_visor_1", "visual_light_1",
                "visual_seam_1", "visual_badge_1", "visual_screw_1", "visual_vent_1")
    for geom in base.findall("geom"):
        name = geom.get("name", "")
        if name.startswith(prefixes) or name == "front":
            material = root.find(f"asset/material[@name='{geom.get('material')}']")
            rgba = material.get("rgba").split()
            rgba[3] = "1" if visible else "0"
            geom.set("rgba", " ".join(rgba))


def chamfered_panel(asset, name, sign):
    # An individual face plate, leaving the rotor bay hollow even visually.
    vertices = []
    for x, cut in ((sign * .072, .006), (sign * .080, .012)):
        for y, z in ((-.070, -.100), (.070, -.100),
                     (.110, .040), (.110, .180 - cut),
                     (.110 - cut, .180), (-.110 + cut, .180),
                     (-.110, .180 - cut), (-.110, .040)):
            vertices.extend((x, y, z))
    ET.SubElement(asset, "mesh", name=name,
                  vertex=" ".join(f"{v:.8g}" for v in vertices))


def apply_appearance(root):
    asset = root.find("asset")
    materials = {
        "ceramic": (".82 .85 .82 1", ".25", ".35"),
        "graphite": (".055 .075 .085 1", ".35", ".45"),
        "alloy": (".42 .50 .53 1", ".6", ".65"),
        "teal": (".025 .62 .56 1", ".45", ".55"),
        "rubber": (".025 .031 .035 1", ".05", ".1"),
        "glass": (".025 .09 .11 1", ".7", ".8"),
    }
    for name, (rgba, specular, shininess) in materials.items():
        ET.SubElement(asset, "material", name=f"design_{name}", rgba=rgba,
                      specular=specular, shininess=shininess)
    ET.SubElement(asset, "material", name="design_light", rgba=".12 .9 .78 1",
                  emission=".6", specular=".3")
    base = root.find("worldbody/body[@name='base']")
    for sign, panel in ((-1, "body"), (1, "body_front")):
        base.find(f"geom[@name='{panel}']").set("rgba", "0 0 0 0")
        chamfered_panel(asset, f"shell_{sign}", sign)
        cosmetic(base, f"shell_{sign}", type="mesh", mesh=f"shell_{sign}", material="design_ceramic")
        face = sign * .0804
        cosmetic(base, f"visor_{sign}", type="box", size=".0005 .075 .013",
                 pos=f"{face} 0 .151", material="design_glass")
        cosmetic(base, f"light_{sign}", type="capsule", size=".0014",
                 fromto=f"{face + sign * .0006} -.045 .151 {face + sign * .0006} .045 .151",
                 material="design_light")
        cosmetic(base, f"seam_{sign}", type="box", size=".0003 .096 .00065",
                 pos=f"{face} 0 .104", material="design_graphite")
        for z, offset in ((-.077, .057), (.080, .087)):
            for y in (-offset, offset):
                cosmetic(base, f"screw_{sign}_{y}_{z}", type="cylinder", size=".0022 .0004",
                         pos=f"{face} {y} {z}", euler="0 1.57079632679 0", material="design_alloy")
        # A small service panel below the equipment seam.
        cosmetic(base, f"badge_{sign}", type="box", size=".0004 .018 .010",
                 pos=f"{face} 0 .049", material="design_teal")
        for i in range(5):
            cosmetic(base, f"vent_{sign}_{i}", type="box", size=".00035 .035 .001",
                     pos=f"{face} 0 {-.053 - i * .005}", material="design_graphite")
    for name in ("body_left", "body_right", "lower_left", "lower_right", "body_floor", "bay_divider"):
        base.find(f"geom[@name='{name}']").set("material", "design_ceramic")
    for name in ("spine", "support_0", "support_1", "fork_tip_0", "fork_tip_1", "rotor_bearing"):
        geom = base.find(f"geom[@name='{name}']")
        geom.set("rgba", ".15 .20 .22 1")
        geom.set("material", "design_alloy")
    for name in ("cargo_deck", "cargo_rail_front", "cargo_rail_rear", "cargo_rail_left", "cargo_rail_right", "equipment", "front"):
        geom = base.find(f"geom[@name='{name}']")
        geom.set("rgba", ".055 .075 .085 1")
        geom.set("material", "design_graphite")
    # Recessed tray surface and tie-down position markings; no cargo is added.
    cosmetic(base, "tray_mat", type="box", size=".076 .106 .0003",
             pos="0 0 .1903", material="design_rubber")
    for index in range(5):
        cosmetic(base, f"tray_rib_{index}", type="box", size=".001 .096 .0002",
                 pos=f"{-.052 + index * .026} 0 .1907", material="design_alloy")
    for y in (-.1125, .1125):
        cosmetic(base, f"side_accent_{y}", type="box", size=".050 .0005 .0015",
                 pos=f"0 {y} .200", material="design_teal")
    for x in (-.082, .082):
        for y in (-.085, .085):
            cosmetic(base, f"tie_mark_{x}_{y}", type="box", size=".002 .008 .0003",
                     pos=f"{x} {y} .2103", material="design_teal")

    drive = base.find("body[@name='drive_wheel']")
    for geom in drive.findall("geom"):
        if geom.get("name", "").startswith("rim_"):
            index = int(geom.get("name").split("_")[1])
            accent = index % 32 < 2
            geom.set("rgba", ".025 .62 .56 1" if accent else ".22 .28 .30 1")
            geom.set("material", "design_teal" if accent else "design_alloy")
        else:
            geom.set("rgba", ".025 .031 .035 1")
            geom.set("material", "design_rubber")

    reaction = base.find("body[@name='reaction_wheel']")
    rotor = reaction.find("geom[@name='flywheel']")
    rotor.set("rgba", ".025 .62 .56 1")
    rotor.set("material", "design_teal")
    hub = reaction.find("geom[@name='flywheel_hub']")
    hub.set("rgba", ".42 .50 .53 1")
    hub.set("material", "design_alloy")
    for sign in (-1, 1):
        cosmetic(reaction, f"rotor_face_{sign}", type="cylinder", size=".075 .00025",
                 pos=f"{sign * .0113} 0 0", euler="0 1.57079632679 0", material="design_graphite")
        cosmetic(reaction, f"hub_cap_{sign}", type="cylinder", size=".014 .0005",
                 pos=f"{sign * .0176} 0 0", euler="0 1.57079632679 0", material="design_graphite")
    cosmetic(reaction, "rotor_tick", type="capsule", size=".0014",
             fromto=".0118 0 .050 .0118 0 .071", material="design_ceramic")
    set_cover_visible(root)

