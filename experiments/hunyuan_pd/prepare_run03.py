"""Generate 96 distinct scene briefs and freeze the four single-run AR points.

Requires only tokenizers, not model weights/torch. Output is never overwritten.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import statistics
from pathlib import Path

from audit_tokens import load_tokenizer, tokenizer_metadata

ROOT = Path(__file__).resolve().parent

# Each brief has three Chinese views and one English view. Views differ near
# the start of the user text; shared instructions occur after that divergence.
SCENES = [
    (
        ("陶器工坊"),
        (
            "靠窗的木工作台上，陶艺师正在修整一只青灰色花瓶，旁边是湿润陶土、竹刀与尚未上釉的杯子。窗外院落有一棵石榴树。"
        ),
        ("pottery studio"),
        (
            "A potter trims a blue gray vase at a wooden bench beside a window. Damp clay, bamboo "
            "tools and unglazed cups sit nearby; a pomegranate tree grows in the courtyard."
        ),
    ),
    (
        ("潮间带考察"),
        (
            "海洋研究员蹲在退潮后的岩石旁观察小水洼，透明水中有海螺、海藻与细沙，防水记录板放在干燥岩面上，远处是平静海湾。"
        ),
        ("tidal pool survey"),
        (
            "A marine researcher examines a shallow pool between coastal rocks at low tide. Snails"
            ", seaweed and sand remain visible through clear water; a waterproof notebook rests ab"
            "ove the waterline."
        ),
    ),
    (
        ("山间茶园"),
        ("采茶人沿着弯曲的小径整理竹篮中的嫩叶，层层茶垄顺着山坡延伸，坡脚有一座旧木棚，薄雾停留在远处的山谷。"),
        ("hillside tea garden"),
        (
            "A tea picker arranges fresh leaves in a bamboo basket on a winding path. Rows of tea "
            "bushes follow the hillside, with a wooden shelter below and mist over the distant val"
            "ley."
        ),
    ),
    (
        ("钟表修理室"),
        (
            "修表师在台灯下检查拆开的机械怀表，铜制齿轮按大小摆在绒布上，镊子、放大镜和玻璃防尘罩各有位置，后方是装满零件的小抽屉。"
        ),
        ("watch repair room"),
        (
            "A watchmaker examines an open mechanical pocket watch under a desk lamp. Brass gears "
            "lie on felt beside tweezers, a magnifier and a glass cover, backed by drawers of spar"
            "e parts."
        ),
    ),
    (
        ("旧书装订"),
        (
            "装订师正在给一本旧书缝制书脊，米色棉线穿过纸页，工作台上放着压书木板、剪刀和不同纹理的封面布，窗台摆着小盆蕨类。"
        ),
        ("bookbinding workshop"),
        (
            "A bookbinder stitches the spine of an old volume with cream cotton thread. Wooden pre"
            "sses, scissors and textured cover cloth lie on the desk; a small fern sits on the win"
            "dowsill."
        ),
    ),
    (
        ("雨后菜市场"),
        ("摊主把带水珠的番茄和茄子放进木筐，遮雨棚边缘仍在滴水，石板路上映着柔和天光，一辆自行车靠在远处的砖墙旁。"),
        ("market after rain"),
        (
            "A market seller arranges wet tomatoes and eggplants in wooden crates. Water drips fro"
            "m the awning, soft daylight reflects on stone paving, and a bicycle leans against a d"
            "istant brick wall."
        ),
    ),
    (
        ("雪地观测站"),
        (
            "气象员在木屋外检查积雪标尺，厚手套握着记录夹，风向旗与小型百叶箱位于屋侧，脚印从门口延伸到仪器附近，远山轮廓清晰。"
        ),
        ("snow weather station"),
        (
            "A weather observer checks a snow gauge outside a timber cabin. A wind flag and instru"
            "ment shelter stand nearby, while footprints connect the door to the gauge beneath dis"
            "tant mountains."
        ),
    ),
    (
        ("面包烘焙间"),
        (
            "烘焙师把发酵好的面团移到木板上，面粉薄薄散落，藤编醒发篮与亚麻布放在一侧，烤炉门后有温暖光线，窗边摆着一束麦穗。"
        ),
        ("bread bakery"),
        (
            "A baker moves risen dough onto a flour dusted wooden board. Proofing baskets and line"
            "n towels stand beside the oven, with a small bundle of wheat near the window."
        ),
    ),
    (
        ("城市屋顶花园"),
        (
            "园艺师在屋顶给香草苗换盆，陶盆、铲子与盛水的金属壶整齐摆在矮台上，攀援植物沿网架生长，远处楼房被午后的薄云衬托。"
        ),
        ("rooftop herb garden"),
        (
            "A gardener repots herbs on a city roof. Terracotta pots, a trowel and a metal waterin"
            "g can occupy a low bench; climbing plants frame the skyline under afternoon clouds."
        ),
    ),
    (
        ("河岸写生"),
        (
            "画者坐在河岸石阶上调配水彩，画纸只露出淡淡的桥梁轮廓，调色盘与水杯放在布垫上，小船系在岸边，河对岸有白墙民居。"
        ),
        ("riverside painting"),
        (
            "An artist mixes watercolors on riverside steps. A faint bridge sketch lies beside a p"
            "alette and water cup, while a small moored boat and white houses appear across the ri"
            "ver."
        ),
    ),
    (
        ("木船维修"),
        (
            "船匠在浅滩旁打磨一条木船的船舷，刨花落在干净的木板上，绳索盘成圆圈，旧船桨靠在支架边，芦苇与缓慢水流构成背景。"
        ),
        ("wooden boat repair"),
        (
            "A boatbuilder sands the side of a wooden hull beside a shallow inlet. Curled shavings"
            " lie on planks; coiled ropes and an old oar rest near reeds and slow water."
        ),
    ),
    (
        ("天文台准备"),
        (
            "观测员在圆顶天文台内校准望远镜，打开的工具包放在侧桌上，弧形轨道和机械转轴清晰可见，圆顶缝隙外露出暮色天空与第一颗亮星。"
        ),
        ("observatory preparation"),
        (
            "An observer aligns a telescope inside a domed observatory. An open tool bag rests bes"
            "ide curved tracks and mechanical pivots; twilight and a bright star show through the "
            "dome slit."
        ),
    ),
    (
        ("玻璃温室"),
        (
            "植物学家在温室中观察一株开花兰花，透明花盆露出部分根系，喷雾瓶和标签夹放在网格台面上，玻璃顶棚挂着细小水珠，后方叶片层次丰富。"
        ),
        ("botanical greenhouse"),
        (
            "A botanist studies a flowering orchid in a greenhouse. Roots show through a clear pot"
            ", a misting bottle sits on a mesh bench, and condensation beads on the glass roof."
        ),
    ),
    (
        ("石桥晨练"),
        (
            "一位练太极的长者站在石桥旁的空地上，宽松衣袖随着动作自然下垂，路边有长椅和落叶，浅雾笼罩着河面，桥拱倒影保持完整。"
        ),
        ("morning beside a stone bridge"),
        (
            "An older adult practices tai chi beside a stone bridge. Loose sleeves follow the move"
            "ment; a bench and fallen leaves border the path while mist softens the river reflecti"
            "on."
        ),
    ),
    (
        ("乐器制作"),
        (
            "制琴师正在检查一把未上漆的小提琴，木屑和弧形模板放在工作台上，墙边悬挂不同尺寸的夹具，侧窗照亮木材细密纹理。"
        ),
        ("violin making"),
        (
            "A luthier inspects an unfinished violin at a workbench. Wood shavings and curved temp"
            "lates lie nearby, clamps hang on the wall, and side window light reveals the grain."
        ),
    ),
    (
        ("果园采收"),
        (
            "果农把刚摘下的梨放进铺有软布的篮子，树叶间仍能看见几个成熟果实，折叠梯靠在树干旁，草地上有斑驳光影，远处是低矮篱笆。"
        ),
        ("orchard harvest"),
        (
            "An orchard worker places fresh pears into a cloth lined basket. Ripe fruit remains am"
            "ong the leaves, a folding ladder leans nearby, and dappled light crosses grass beside"
            " a low fence."
        ),
    ),
    (
        ("灯塔值守"),
        (
            "值守员在灯塔入口整理航海绳结，厚玻璃窗透出暖光，墙脚有防水靴与金属提灯，海浪拍打远处礁石，天空呈柔和蓝灰色。"
        ),
        ("lighthouse keeper"),
        (
            "A keeper sorts nautical rope knots at a lighthouse entrance. Warm light shines throug"
            "h thick glass, with boots and a metal lantern beside the wall and waves breaking on d"
            "istant rocks."
        ),
    ),
    (
        ("织布工作室"),
        ("织工在木织机前整理彩色经线，梭子穿过整齐的线束，几卷植物染线放在矮凳上，墙上挂着简洁织样，窗帘轻轻垂落。"),
        ("weaving studio"),
        (
            "A weaver arranges colored warp threads on a wooden loom. A shuttle passes between the"
            " strands, naturally dyed yarn rests on a stool, and simple woven samples hang beside "
            "a curtain."
        ),
    ),
    (
        ("山村邮递"),
        (
            "邮递员在村口检查帆布邮袋，一辆装有货架的自行车停在旁边，石阶通向木门，墙角有盛开的野花，蜿蜒道路连接远处山坡上的房屋。"
        ),
        ("village mail delivery"),
        (
            "A mail carrier checks a canvas bag beside a bicycle with a rear rack. Stone steps lea"
            "d to a wooden doorway, wildflowers grow by the wall, and a winding road links hillsid"
            "e homes."
        ),
    ),
    (
        ("博物馆修复"),
        (
            "修复师戴着薄手套清理一件陶瓷器物的边缘，软刷和小托盘放在无尘台上，台面覆盖浅灰保护垫，背景玻璃柜内陈列几件完整器物。"
        ),
        ("museum conservation"),
        (
            "A conservator wearing thin gloves cleans the rim of a ceramic vessel. Soft brushes an"
            "d trays sit on a gray protective mat, with intact objects displayed in a glass cabine"
            "t behind."
        ),
    ),
    (
        ("港口渔网"),
        (
            "渔民坐在码头边修补绿色渔网，浮球和线轴摆在木箱上，系泊缆绳连接船身与岸桩，远处有平静水面和低矮仓库，海鸟停在栏杆上。"
        ),
        ("harbor net mending"),
        (
            "A fisher repairs a green net beside the quay. Floats and thread spools lie on a crate"
            ", mooring lines connect a boat to bollards, and a seabird rests above the calm harbor"
            "."
        ),
    ),
    (
        ("咖啡烘焙"),
        (
            "咖啡师在小型烘焙机旁观察豆子的颜色，金属托盘装着不同烘焙程度的咖啡豆，麻袋与秤放在背景，通风窗照进柔和日光，工作区整洁。"
        ),
        ("coffee roasting room"),
        (
            "A roaster studies coffee beans beside a compact roasting machine. Metal trays hold se"
            "veral roast shades, sacks and scales occupy the background, and daylight enters throu"
            "gh a ventilation window."
        ),
    ),
    (
        ("自然标本整理"),
        (
            "研究者在木桌上整理落叶与松果标本，放大镜下能看见清晰叶脉，纸袋与细刷摆在一侧，窗外是树林，桌边的布袋露出采集工具。"
        ),
        ("natural history collection"),
        (
            "A researcher arranges fallen leaves and pinecones on a wooden table. A magnifier reve"
            "als veins, paper bags and brushes sit nearby, and collecting tools protrude from a ca"
            "nvas bag."
        ),
    ),
    (
        ("铁路模型制作"),
        (
            "模型师正在微调微缩铁路上的站台位置，小镊子夹住一盏路灯，桌面有轨道、树木模型和未完成的车站，背景工具架整齐，灯光柔和。"
        ),
        ("miniature railway workshop"),
        (
            "A model maker adjusts a platform on a miniature railway. Tweezers hold a tiny lamp be"
            "side tracks, model trees and an unfinished station, with neatly arranged tools in sof"
            "t light."
        ),
    ),
]

ZH_DETAIL = (
    "请将这些元素组织成一张完整、自然的环境人物照片，使用实际1024×1024的正方形画幅，不添加边框、拼贴或画中画。人物与正在进行的活动是视觉重点，环境提供明确的地点信息，"
    "但不要让背景装饰抢占主体。保留场景中描述的工具和物品，把它们放在合理的支撑面上，避免物件漂浮、重复或相互穿透。人物只做描述中的一项主要动作，手指与工具接触位置准确，衣服褶"
    "皱随姿态变化，身体比例自然。保持所有人物为成年人，不增添围观人群。\n画面需要清楚区分前景、中景和背景。前景使用少量与场景有关的材质或工具引导视线，中景保留完整主体轮廓，背"
    "景通过距离和适度虚化形成纵深；不要用过强虚化遮盖空间关系。地面与台面的透视应互相一致，门窗、支架和垂直结构保持合理方向，物品大小与人物尺度协调。让主体四周留出适当空间，避"
    "免手、工具尖端或头顶被画幅边缘意外切断。\n光线应来自场景内可以解释的方向，保留明暗过渡与柔和接触阴影。主光照亮主要动作和关键物件，背光只用于分离轮廓，不制造夸张光环。金属"
    "可以有细小而克制的高光，玻璃保持透明，木材、布料、纸张与植物表面呈现各自不同的粗糙度；不要把所有物体处理成相同的塑料质感。暗部依然可辨认细节，亮部不过曝，肤色与环境反射保"
    "持自然。\n采用细腻的纪实摄影观感，色彩以该场景原有的自然颜色为主，只在少数工具或衣物上保留适量强调色。画面清晰但不要出现过度锐化的白边，也不要加入浓重滤镜、人工炫光或不必"
    "要的运动拖影。允许真实使用痕迹，例如轻微磨损、折痕、湿润表面和不完全规则的排列，使环境看起来正在被使用，而不是无人接触的商品展示。不要生成可读文字、标志、字幕、水印或品牌"
    "名称。最终画面应让观察者一眼辨认场景、人物动作与空间层次。"
)

EN_DETAIL = (
    "Create one coherent environmental documentary photograph on an actual 1024 by 1024 sq"
    "uare canvas. The person and the described activity are the main visual subject. Keep "
    "all people adults. Use only the principal action already described, rather than addin"
    "g competing activities or a crowd. Arrange the named tools and objects on credible su"
    "pporting surfaces, preserving their relationship to the task. Hands should meet handl"
    "es or materials correctly, with natural anatomy and clothing folds that follow the po"
    "se. Leave sufficient space around the head, elbows and working tools so that the fram"
    "e does not accidentally cut through important details.\nBuild a clear foreground, midd"
    "le distance and background. Include only a few relevant foreground objects to guide t"
    "he eye toward the activity. Preserve the complete silhouette of the central subject. "
    "Use a gentle reduction in contrast and moderate background softness to convey depth, "
    "while keeping the location recognizable. Align the perspective of the floor, work sur"
    "faces and architectural elements. Doorways, windows, shelves and supports should have"
    " plausible proportions. Avoid floating objects, duplicated tools, intersecting surfac"
    "es, impossible reflections and exaggerated wide angle distortion. Preserve the practi"
    "cal scale of every object relative to the person.\nUse physically plausible light from"
    " a direction that can be understood from the scene. Illuminate the working hands and "
    "the most important material, with gradual transitions between light and shade. A smal"
    "l amount of reflected light may reveal detail in the shadows, but do not flatten the "
    "whole image. Keep bright surfaces within a natural exposure range. Any rim light shou"
    "ld be subtle rather than a glowing outline. Make contact shadows visible beneath obje"
    "cts and where surfaces meet. Reflections on metal or glass must agree with the surrou"
    "nding space and the direction of illumination.\nDistinguish materials through texture "
    "rather than extreme contrast. Wood should retain its grain and minor wear, fabric sho"
    "uld show its weave and believable folds, and paper should retain a soft matte surface"
    ". Where plants, stone, clay, water or food occur in the brief, preserve their particu"
    "lar physical character. Transparent surfaces should remain transparent, and polished "
    "objects should not resemble mirrors unless their material requires it. Avoid a unifor"
    "m plastic appearance. Include restrained signs of ordinary use such as small scratche"
    "s, creases, damp patches or irregular arrangements, without making the environment di"
    "rty or abandoned.\nKeep a natural color palette drawn from the location and its materi"
    "als. Reserve a limited accent color for a small tool or item of clothing rather than "
    "scattering bright colors throughout the composition. Skin tones should respond natura"
    "lly to reflected environmental light. The image should feel carefully observed, with "
    "enough detail to inspect the activity and surrounding textures, but without halos fro"
    "m excessive sharpening. Do not add artificial lens flares, strong color filters, dram"
    "atic motion streaks, a collage, decorative borders or a separate inset image. Exclude"
    " readable lettering, subtitles, watermarks, brand marks and labels. The final picture"
    " should communicate a specific place, an understandable action and a consistent spati"
    "al arrangement at first glance."
)


def build_catalog(tokenizer):
    source = ROOT.parents[1] / "vllm_omni/diffusion/models/hunyuan_image3/system_prompt.py"
    spec = importlib.util.spec_from_file_location("hy3_system_prompt", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Same segmented t2i/think template as prompt_utils.build_prompt_tokens.
    overhead = 2 + sum(
        len(tokenizer.encode(s, add_special_tokens=False).ids)
        for s in (
            module.get_system_prompt("en_unified", "think").strip(),
            "\n\n",
            "User: ",
            "\n\nAssistant: ",
        )
    )
    rows = []
    views = ("从斜侧方拍摄工作中的", "通过近处物件构图呈现", "以略高视点记录")
    for i, (title, brief, en_title, en_brief) in enumerate(SCENES):
        for view in range(4):
            language = "en" if view == 3 else "zh"
            prompt = (
                f"{en_title.capitalize()}: {en_brief}\n"
                f"Use an eye-level medium-wide view with a calm, observational mood.\n{EN_DETAIL}"
                if language == "en"
                else f"{views[view]}{title}：{brief}\n{ZH_DETAIL}"
            )
            if language == "zh":
                prompt += (
                    f"\n在{title}的环境中安排一条清楚的视觉阅读路径：先看到人物的主要动作，再看到与动作直接有关的物件，"
                    "最后注意到远处环境。使用自然的遮挡表现物体前后关系，不要让背景轮廓恰好穿过人物头部或手部。"
                    "人物的视线应落在正在处理的对象上，表情专注而放松，肩膀与手臂姿态符合动作需要。"
                    "选择适合这项活动的普通实用服装，颜色低调，衣料厚薄与环境相符，不使用戏剧化服饰或舞台化妆。"
                    "画面中的细节应服务于具体场景，不加入与活动无关的装饰品；让常用工具靠近手边，备用物品稍远，通行位置保持合理空间。"
                    "如果存在水面或潮湿物体，反射应具有正确方向和适当清晰度；如果存在植物，叶片朝向和枝条连接应自然。"
                    "将这一刻表现为真实活动中的短暂片段，而不是面向镜头刻意摆出的宣传照。"
                )
            else:
                prompt += (
                    f"\nWithin this {en_title}, establish a clear visual route from the person's main action "
                    "to the tool or material involved, then to the surroundings. Avoid background edges crossing "
                    "the head or hands. Direct the person's gaze toward the task, with a focused but relaxed "
                    "expression and a practical stance. Use ordinary working clothes suited to the environment, "
                    "without theatrical costume or makeup. Place frequently used tools within reach, with spare "
                    "materials farther away and a credible route through the working area. Treat this as a briefly "
                    "observed moment in an ongoing activity, rather than "
                    "a promotional pose directed at the camera."
                )
            count = len(tokenizer.encode(prompt, add_special_tokens=False).ids)
            rows.append(
                dict(
                    id=f"run03_scene{i + 1:02d}_view{view + 1}",
                    language=language,
                    group="long",
                    workload="single" if i < 12 else "waves",
                    user_tokens=count,
                    full_input_tokens=count + overhead,
                    prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                    request=dict(
                        prompt=prompt,
                        width=1024,
                        height=1024,
                        size="1024x1024",
                        n=1,
                        num_inference_steps=50,
                        guidance_scale=0.0,
                        seed=42,
                        bot_task="think",
                        use_system_prompt="en_unified",
                    ),
                )
            )
    return rows


def subset(rows):
    selected = []
    for language in ("zh", "en"):
        pool = sorted((r for r in rows if r["language"] == language), key=lambda r: (r["full_input_tokens"], r["id"]))
        target = statistics.mean(r["full_input_tokens"] for r in pool)
        choices = (pool[::2], pool[1::2])
        selected.extend(min(choices, key=lambda xs: abs(statistics.mean(r["full_input_tokens"] for r in xs) - target)))
    ids = {r["id"] for r in selected}
    return [r for r in rows if r["id"] in ids]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def prepare(output, tokenizer_path, delay=5.0):
    if not math.isfinite(delay) or delay <= 0:
        raise ValueError("Wave delay must be finite and positive")
    tokenizer = load_tokenizer(tokenizer_path)
    rows = build_catalog(tokenizer)
    assert len(rows) == len({r["prompt_sha256"] for r in rows}) == 96
    output.mkdir(parents=True, exist_ok=False)
    points = []
    for workload in ("single", "waves"):
        parent = [r for r in rows if r["workload"] == workload]
        waves = [parent] if workload == "single" else [parent[:24], parent[24:]]
        for deployment in ("pd", "npd"):
            selected = [wave if deployment == "pd" else subset(wave) for wave in waves]
            frozen = [
                dict(row, wave_id=i + 1, scheduled_offset_s=0.0 if i == 0 else delay)
                for i, wave in enumerate(selected)
                for row in wave
            ]
            name = f"{deployment}-{workload}.jsonl"
            raw = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in frozen).encode()
            (output / name).write_bytes(raw)
            points.append(
                dict(
                    deployment=deployment,
                    workload=workload,
                    path=name,
                    sha256=hashlib.sha256(raw).hexdigest(),
                    requests=len(frozen),
                    wave_sizes=[len(wave) for wave in selected],
                    full_input_tokens_mean=statistics.mean(r["full_input_tokens"] for r in frozen),
                )
            )
    manifest = dict(
        version=3,
        rounds=1,
        max_num_batched_tokens=8192,
        ar_max_num_seqs=64,
        wave_delay_s=delay,
        tokenizer=tokenizer_metadata(tokenizer_path),
        points=points,
        token_counts="Offline t2i/think/en_unified segmented template; verify against runtime audit",
        unique_prompts=96,
        input_tokens_range=[min(r["full_input_tokens"] for r in rows), max(r["full_input_tokens"] for r in rows)],
    )
    write_json(output / "manifest.json", manifest)
    preview = ["Run03 frozen prompts (single and waves use disjoint scenes)", ""]
    for row in rows:
        preview.extend(
            [
                f"{row['id']} | {row['workload']} | {row['language']} | full tokens={row['full_input_tokens']}",
                "",
                row["request"]["prompt"],
                "",
            ]
        )
    (output / "prompts.md").write_text("\n".join(preview), encoding="utf-8", newline="\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--tokenizer-json", required=True, type=Path)
    parser.add_argument("--wave-delay", type=float, default=5.0)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output_dir, args.tokenizer_json, args.wave_delay), ensure_ascii=False, indent=2))
