from __future__ import annotations

import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_ROOT / "assets" / "mujoco_menagerie" / "franka_emika_panda"
OUTPUT_DIR = PROJECT_ROOT / "models" / "franka_emika_panda_torque"
PANDA_XML = "panda.xml"

HOME_QPOS = "0 0 0 -1.57079 0 1.57079 -0.7853 0.04 0.04"
HOME_CTRL = "0 0 0 0 0 0 0 255"
TORQUE_LIMITS = [87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0]
EE_SITE_NAME = "attachment_site"


def _replace_arm_actuators(xml_path: Path) -> None:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    actuator = root.find("actuator")
    if actuator is None:
        raise RuntimeError(f"No <actuator> section found in {xml_path}")

    original = list(actuator)
    if len(original) < 8:
        raise RuntimeError(f"Expected at least 8 actuators in {xml_path}, got {len(original)}")

    gripper = original[7]
    actuator.clear()

    for index, limit in enumerate(TORQUE_LIMITS, start=1):
        motor = ET.Element(
            "motor",
            {
                "name": f"torque{index}",
                "joint": f"joint{index}",
                "gear": "1",
                "ctrllimited": "true",
                "ctrlrange": f"{-limit:g} {limit:g}",
                "forcelimited": "true",
                "forcerange": f"{-limit:g} {limit:g}",
            },
        )
        actuator.append(motor)

    actuator.append(gripper)

    keyframe = root.find("keyframe")
    if keyframe is None:
        raise RuntimeError(f"No <keyframe> section found in {xml_path}")
    home = keyframe.find("./key[@name='home']")
    if home is None:
        raise RuntimeError(f"No home keyframe found in {xml_path}")
    home.set("qpos", HOME_QPOS)
    home.set("ctrl", HOME_CTRL)

    hand = root.find(".//body[@name='hand']")
    if hand is None:
        raise RuntimeError(f"No hand body found in {xml_path}; cannot add EE site")
    existing_site = root.find(f".//site[@name='{EE_SITE_NAME}']")
    if existing_site is None:
        ET.SubElement(
            hand,
            "site",
            {
                "name": EE_SITE_NAME,
                "pos": "0 0 0.1",
                "size": "0.001",
                "rgba": "0.5 0.5 0.5 0.3",
                "group": "4",
            },
        )

    ET.indent(tree, space="  ")
    tree.write(xml_path, encoding="utf-8", xml_declaration=False)


def prepare_torque_model() -> Path:
    source_xml = SOURCE_DIR / PANDA_XML
    if not source_xml.exists():
        raise FileNotFoundError(f"Official Panda XML not found: {source_xml}")

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    shutil.copytree(SOURCE_DIR, OUTPUT_DIR)

    output_xml = OUTPUT_DIR / PANDA_XML
    _replace_arm_actuators(output_xml)

    print(f"Source path: {SOURCE_DIR}")
    print(f"Output path: {OUTPUT_DIR}")

    try:
        model = mujoco.MjModel.from_xml_path(str(output_xml))
    except Exception as exc:
        raise RuntimeError(f"Failed to compile derived XML: {output_xml}") from exc

    print(f"Compiled derived model successfully: nq={model.nq}, nv={model.nv}, nu={model.nu}")
    return output_xml


if __name__ == "__main__":
    try:
        prepare_torque_model()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise
