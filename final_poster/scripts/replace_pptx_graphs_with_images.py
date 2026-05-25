from __future__ import annotations

import shutil
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "final_research_poster_A1_portrait_english.pptx"
OUTPUT = ROOT / "final_research_poster_A1_portrait_english_image_graphs.pptx"
FIGURE_DIR = ROOT / "figures" / "poster_main"

NS = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "rel": "http://schemas.openxmlformats.org/package/2006/relationships",
    "ct": "http://schemas.openxmlformats.org/package/2006/content-types",
}

for prefix in ("a", "p", "r"):
    ET.register_namespace(prefix, NS[prefix])
ET.register_namespace("", NS["rel"])


def q(ns: str, tag: str) -> str:
    return f"{{{NS[ns]}}}{tag}"


def find_sp_tree(slide_root: ET.Element) -> ET.Element:
    sp_tree = slide_root.find(".//p:spTree", NS)
    if sp_tree is None:
        raise RuntimeError("slide1.xml has no p:spTree")
    return sp_tree


def shape_id(element: ET.Element) -> int | None:
    c_nv_pr = element.find(".//p:cNvPr", NS)
    if c_nv_pr is None:
        return None
    value = c_nv_pr.get("id")
    return int(value) if value and value.isdigit() else None


def remove_id_ranges(sp_tree: ET.Element, ranges: list[range]) -> dict[int, int]:
    children = list(sp_tree)
    child_ids = [shape_id(child) for child in children]
    remove_ids = {sid for sid in child_ids if sid is not None and any(sid in r for r in ranges)}
    remove_ids.discard(None)

    removed_insert_positions: dict[int, int] = {}
    kept_before = 0
    for child, sid in zip(children, child_ids):
        for id_range in ranges:
            if sid == id_range.start:
                removed_insert_positions[id_range.start] = kept_before
                break
        if sid not in remove_ids:
            kept_before += 1

    for child, sid in zip(children, child_ids):
        if sid in remove_ids:
            sp_tree.remove(child)
    return removed_insert_positions


def contain_box(path: Path, x: int, y: int, w: int, h: int) -> tuple[int, int, int, int]:
    with Image.open(path) as image:
        img_w, img_h = image.size
    scale = min(w / img_w, h / img_h)
    new_w = int(round(img_w * scale))
    new_h = int(round(img_h * scale))
    new_x = x + (w - new_w) // 2
    new_y = y + (h - new_h) // 2
    return new_x, new_y, new_w, new_h


def make_pic(shape_id_value: int, name: str, rel_id: str, x: int, y: int, w: int, h: int) -> ET.Element:
    pic = ET.Element(q("p", "pic"))

    nv_pic_pr = ET.SubElement(pic, q("p", "nvPicPr"))
    c_nv_pr = ET.SubElement(nv_pic_pr, q("p", "cNvPr"))
    c_nv_pr.set("id", str(shape_id_value))
    c_nv_pr.set("name", name)
    c_nv_pic_pr = ET.SubElement(nv_pic_pr, q("p", "cNvPicPr"))
    ET.SubElement(c_nv_pic_pr, q("a", "picLocks"), {"noChangeAspect": "1"})
    ET.SubElement(nv_pic_pr, q("p", "nvPr"))

    blip_fill = ET.SubElement(pic, q("p", "blipFill"))
    ET.SubElement(blip_fill, q("a", "blip"), {q("r", "embed"): rel_id})
    stretch = ET.SubElement(blip_fill, q("a", "stretch"))
    ET.SubElement(stretch, q("a", "fillRect"))

    sp_pr = ET.SubElement(pic, q("p", "spPr"))
    xfrm = ET.SubElement(sp_pr, q("a", "xfrm"))
    ET.SubElement(xfrm, q("a", "off"), {"x": str(x), "y": str(y)})
    ET.SubElement(xfrm, q("a", "ext"), {"cx": str(w), "cy": str(h)})
    prst_geom = ET.SubElement(sp_pr, q("a", "prstGeom"), {"prst": "rect"})
    ET.SubElement(prst_geom, q("a", "avLst"))

    return pic


def read_xml(zf: zipfile.ZipFile, name: str) -> ET.Element:
    return ET.fromstring(zf.read(name))


def write_xml(root: ET.Element) -> bytes:
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def add_png_content_type(content_types: ET.Element) -> None:
    has_png = any(
        child.tag == q("ct", "Default")
        and child.get("Extension", "").lower() == "png"
        for child in content_types
    )
    if not has_png:
        content_types.insert(0, ET.Element(q("ct", "Default"), {"Extension": "png", "ContentType": "image/png"}))


def add_image_relationships(rels_root: ET.Element, rels: list[tuple[str, str]]) -> None:
    for rel_id, media_name in rels:
        rel = ET.Element(q("rel", "Relationship"))
        rel.set("Id", rel_id)
        rel.set("Type", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image")
        rel.set("Target", f"../media/{media_name}")
        rels_root.append(rel)


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)

    required_images = [
        "doorkey6x6_3000_mean_success.png",
        "multiroom_n4s4_auc_and_speed.png",
        "multiroom_n4s4_final_episode_length.png",
        "hidden_intervention_success_drop.png",
        "manager_goal_policy_coupling.png",
        "multiroom_n4s5_reward_signal_boundary.png",
    ]
    missing = [name for name in required_images if not (FIGURE_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing figure(s): {', '.join(missing)}")

    # Target boxes are the original drawn-graph regions in slide EMUs.
    replacements = [
        {
            "start_id": 85,
            "image": "doorkey6x6_3000_mean_success.png",
            "media": "image_graph_doorkey6x6_3000_mean_success.png",
            "rel_id": "rId3",
            "shape_id": 300,
            "name": "DoorKey-6x6 PNG graph",
            "box": (7679741, 3520440, 6024677, 4517136),
        },
        {
            "start_id": 125,
            "image": "multiroom_n4s4_auc_and_speed.png",
            "media": "image_graph_multiroom_n4s4_auc_and_speed.png",
            "rel_id": "rId4",
            "shape_id": 301,
            "name": "MultiRoom N4-S4 AUC and speed PNG graph",
            "box": (7624877, 13807440, 6134405, 2865120),
        },
        {
            "start_id": 125,
            "image": "multiroom_n4s4_final_episode_length.png",
            "media": "image_graph_multiroom_n4s4_final_episode_length.png",
            "rel_id": "rId5",
            "shape_id": 302,
            "name": "MultiRoom N4-S4 final episode length PNG graph",
            "box": (7624877, 16855440, 6134405, 4309320),
        },
        {
            "start_id": 199,
            "image": "hidden_intervention_success_drop.png",
            "media": "image_graph_hidden_intervention_success_drop.png",
            "rel_id": "rId6",
            "shape_id": 303,
            "name": "Hidden intervention PNG graph",
            "box": (14664538, 3703320, 5896661, 2255520),
        },
        {
            "start_id": 199,
            "image": "manager_goal_policy_coupling.png",
            "media": "image_graph_manager_goal_policy_coupling.png",
            "rel_id": "rId7",
            "shape_id": 304,
            "name": "Manager goal policy coupling PNG graph",
            "box": (14664538, 6199632, 5896661, 4587240),
        },
        {
            "start_id": 228,
            "image": "multiroom_n4s5_reward_signal_boundary.png",
            "media": "image_graph_multiroom_n4s5_reward_signal_boundary.png",
            "rel_id": "rId8",
            "shape_id": 305,
            "name": "MultiRoom N4-S5 boundary PNG graph",
            "box": (14664538, 17693640, 5896661, 4114800),
        },
    ]

    remove_ranges = [range(85, 120), range(125, 169), range(199, 223), range(228, 251)]

    with zipfile.ZipFile(SOURCE, "r") as src:
        slide_root = read_xml(src, "ppt/slides/slide1.xml")
        rels_root = read_xml(src, "ppt/slides/_rels/slide1.xml.rels")
        content_types = read_xml(src, "[Content_Types].xml")

        sp_tree = find_sp_tree(slide_root)
        insert_positions = remove_id_ranges(sp_tree, remove_ranges)

        grouped: dict[int, list[dict[str, object]]] = {}
        for item in replacements:
            grouped.setdefault(int(item["start_id"]), []).append(item)

        offset = 0
        for start_id in sorted(grouped):
            insert_at = insert_positions[start_id] + offset
            for item in grouped[start_id]:
                image_path = FIGURE_DIR / str(item["image"])
                x, y, w, h = contain_box(image_path, *item["box"])  # type: ignore[arg-type]
                pic = make_pic(
                    int(item["shape_id"]),
                    str(item["name"]),
                    str(item["rel_id"]),
                    x,
                    y,
                    w,
                    h,
                )
                sp_tree.insert(insert_at, pic)
                insert_at += 1
                offset += 1

        add_image_relationships(rels_root, [(str(item["rel_id"]), str(item["media"])) for item in replacements])
        add_png_content_type(content_types)

        slide_xml = write_xml(slide_root)
        rels_xml = write_xml(rels_root)
        content_types_xml = write_xml(content_types)

        temp_output = OUTPUT.with_suffix(".tmp.pptx")
        if temp_output.exists():
            temp_output.unlink()

        with zipfile.ZipFile(temp_output, "w", zipfile.ZIP_DEFLATED) as dst:
            for info in src.infolist():
                if info.filename in {
                    "ppt/slides/slide1.xml",
                    "ppt/slides/_rels/slide1.xml.rels",
                    "[Content_Types].xml",
                }:
                    continue
                dst.writestr(info, src.read(info.filename))
            dst.writestr("ppt/slides/slide1.xml", slide_xml)
            dst.writestr("ppt/slides/_rels/slide1.xml.rels", rels_xml)
            dst.writestr("[Content_Types].xml", content_types_xml)
            for item in replacements:
                dst.write(FIGURE_DIR / str(item["image"]), f"ppt/media/{item['media']}")

    if OUTPUT.exists():
        OUTPUT.unlink()
    shutil.move(temp_output, OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
