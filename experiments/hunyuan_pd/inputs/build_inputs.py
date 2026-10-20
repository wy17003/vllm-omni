"""Build paired t2i inputs from local source material; never execute source scripts."""

# ruff: noqa: E501 -- Curated prompt literals retain their exact text for auditing.
from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SPECS = [
    (
        "TXT-001",
        "溪水秋千",
        "zh",
        "一位成年女性穿米白纱裙坐在浅溪中的秋千上，脚尖轻触水面。绿叶与树枝围绕她，溪底石块清晰可见。自然侧光、青绿色调，清新柔和的写实摄影。",
    ),
    (
        "TXT-002",
        "红裙镜面",
        "zh",
        "一位成年女性穿红色花瓣装饰礼裙，在昏暗室内侧身蹲坐，用手机对镜自拍。深棕长发自然垂落，黑色镜框与模糊绿植构成背景，冷调侧光衬托红裙的层次。",
    ),
    (
        "TXT-003",
        "车窗日光",
        "zh",
        "一位成年女性坐在汽车后座，穿淡紫色针织无袖上衣，长发垂肩，神情安静。焦糖棕皮质座椅与窗外翠绿植物形成柔和对比，日光透过车窗，呈现自然抓拍质感。",
    ),
    (
        "TXT-005",
        "角色档案",
        "zh",
        "设计成年角色星野的资料卡：冰川浅蓝薄纱裙、冷茶棕卷发、冰晶发饰，白底与银灰装饰。排入半身特写、正侧背三视图、六个表情头像及配饰细节，所有视图保持角色一致。",
    ),
    (
        "TXT-007",
        "时尚绘画拼贴",
        "zh",
        "一位成年中性模特穿黑色不对称衬衫和画布质感长裙，站在奶油色、炭黑与锈红颜料构成的背景前。粗笔触与褪色肖像叠加，左上强光制造清晰阴影，呈现摄影与抽象绘画融合的时尚拼贴。",
    ),
    (
        "TXT-008",
        "宫廷灯笼",
        "en",
        "An adult Chinese noblewoman in ivory, celadon and gold embroidered Hanfu stands beside a bronze palace lantern. Ornate floral hairpins, turquoise courtyard walls, soft daylight and shallow depth of field create a refined photographic portrait.",
    ),
    (
        "TXT-009",
        "红线刺绣",
        "zh",
        "在天然米色亚麻布上，用绯红色细线绣出一位成年女性的居中肖像。面部、卷发和衣领由清晰针脚塑造，几根松线连接下方木质线轴。暖色侧光突出纤维、凸起和接触阴影，保持真实手工质感。",
    ),
    (
        "TXT-017",
        "树脂动物键盘",
        "en",
        "Photograph a complete mechanical keyboard with transparent resin keycaps containing woodland animal miniatures: foxes, rabbits, squirrels, owls and deer. Preserve readable key legends and realistic keyboard proportions, with natural reflections and detailed macro textures.",
    ),
    (
        "MD-01",
        "复古室内人像",
        "en",
        "An adult Asian woman in a dark teal dress sits on a worn red velvet armchair in a dim vintage room. Patterned wallpaper and projected shadows frame her thoughtful expression. Cinematic lighting reveals fabric textures while keeping the background softly blurred.",
    ),
    (
        "MD-02",
        "街角光影",
        "en",
        "A warm orange stucco building with white stripes stands at a sunlit street corner. Closed dark shutters, a hanging black lamp and rough stone foundations cast sharp shadows. A pedestrian silhouette crosses the pavement in a cinematic, photorealistic composition.",
    ),
    (
        "MD-03",
        "汉服红底特写",
        "zh",
        "一位成年女性穿古风汉服，微微侧头注视镜头，肩部以上近景特写。深红哑光背景保持简洁，暗橘色侧光塑造面部，银白轮廓光勾勒人物边缘，融合古典服饰与现代杂志摄影。",
    ),
    (
        "MD-04",
        "厚涂微缩湿地",
        "zh",
        "白色纹理纸上，一道斜向红色厚涂笔触承载微缩湿地：红海滩、蓝白湖泊、浅黄芦苇和一只白鹭。保留画刀痕迹与颜料起伏，侧光形成细小阴影，大面积留白，极简油画风格。",
    ),
    (
        "MD-05",
        "兔子材质四宫格",
        "zh",
        "用二乘二四宫格展示同一只坐姿兔子：左上白色石膏，右上透明玻璃，左下拉丝钛金属，右下灰色毛绒。姿态、视角与深灰背景一致，柔和棚灯准确表现四种材质，写实三维渲染。",
    ),
    (
        "MD-06",
        "日常四种画风",
        "zh",
        "二乘二四宫格描绘同一个小男孩的一天：左上写实摄影表现清晨上学，右上赛璐璐动漫表现午餐，左下铅笔素描表现踢足球，右下后印象派油画表现星夜河边钓鱼。四格边界整齐，人物身份一致。",
    ),
    (
        "MD-07",
        "鹦鹉素描教程",
        "zh",
        "制作三乘三九宫格鹦鹉素描教程，依次展示几何起形、鸟喙尾巴、轮廓、翅膀栖木、羽毛、初步阴影、细化排线、木纹及完成图。每格右上标注数字1至9，浅灰背景，均匀光线，步骤清晰。",
    ),
    (
        "MD-08",
        "企鹅模型海报",
        "zh",
        "现代简洁海报：红围巾QQ企鹅居中，黑色身体、白色腹部、黄色嘴脚，圆润三维质感。背景从浅蓝渐变至白色，底部两行清晰文字为“Hunyuan Image 3.0”和“原生多模态大模型”，光线均匀。",
    ),
    (
        "TXT-025",
        "银甲武将",
        "zh",
        "一位成年中国武将身穿银色花丝盔甲、象牙白长袍和深蓝斗篷，闭目俯身站立，手持向下的弯刀。长发与丝带迎风飘动，浅灰背景，大幅留白，水墨线条结合精细金属渲染。",
    ),
    (
        "TXT-027",
        "湖畔草地抓拍",
        "en",
        "An adult woman in a cream linen shirt and blue jeans stands on grass beside a lake, carrying a woven straw bag. Sunglasses rest on her black hair. Soft sunlight, distant trees and scattered clouds create a natural eye-level snapshot.",
    ),
    (
        "TXT-028",
        "潮汐石台",
        "zh",
        "一位成年女性穿有完整内衬的白色连衣裙，站在浅潮淹没的古老石台上，微笑着轻提裙摆，水珠在小腿旁飞溅。远处蓝绿色海浪与阴云形成层次，低机位拍摄，柔和日光，写实电影质感。",
    ),
    (
        "TXT-029",
        "桂花白墙海报",
        "en",
        "An adult woman in pale Hanfu gently smells osmanthus beneath delicate branches beside a warm white wall. Sparse yellow flowers, muted green leaves and soft leaf shadows form a minimalist autumn poster, with the title Early Autumn in the clear space.",
    ),
    (
        "TXT-031",
        "青柠跑车海报",
        "zh",
        "青柠绿保时捷911 GT3位于现代图形海报中央，四分之三侧视，大尾翼与黑色轮毂清晰。白色背景配青柠竖条，上方品牌名、下方车型名，明亮均匀照明，汽车与排版保持完整。",
    ),
    (
        "TXT-033",
        "微缩咖啡广告",
        "zh",
        "一位穿粉色上衣的成年微缩女性坐在巨大透明草莓拿铁杯盖边缘，旁边是金链粉色手提包与心形太阳镜。蓝色咖啡品牌标志清晰，户外路面与自然柔光，所有物件完整入镜。",
    ),
    (
        "TXT-034",
        "柠檬汽水飞溅",
        "zh",
        "黄绿色柠檬汽水罐悬于深绿渐变背景前，罐身凝结水珠，柠檬切片、绿叶与果汁飞溅环绕。高速摄影凝固水滴，棚灯勾勒金属高光和透明果肉，清爽写实的饮品广告。",
    ),
    (
        "TXT-036",
        "笔记本立体人物",
        "zh",
        "木桌上的螺旋笔记本中，一位卡通人物仿佛从纸页走出。下半身保留铅笔线稿，上半身变为立体形象，棕色卷发、柔色上衣、灰色牛仔裤与白鞋，柔和侧光突出纸面与立体阴影。",
    ),
    (
        "TXT-038",
        "足球珐琅徽章",
        "zh",
        "黑色牛仔布上别着一枚蓝星足球队珐琅徽章，金色边框、深蓝底色与白色足球图案清晰。微距展示表面细小磨痕、针孔和织物纤维，背景柔和虚化，影棚主灯形成真实金属反光。",
    ),
    (
        "TXT-043",
        "海岸旅行自拍",
        "zh",
        "一位成年女性在浅海与沙滩交界处伸手自拍，穿白色钩针泳衣和淡粉花卉罩衫，深棕发丝被海风吹动，露齿微笑。青绿色海水、弯曲沙滩与薄云同框，明亮日光和自然手机广角质感。",
    ),
    (
        "TXT-044",
        "夏日西瓜",
        "zh",
        "成年女性穿淡紫花卉连衣裙、戴绿色太阳镜，在海滩上双手握木棒压向裂开的西瓜，开心大笑。西瓜和水滴位于前景，蓝白休闲垫、碧海与蓝天构成背景，阳光明亮，高速快门凝固动感。",
    ),
    (
        "TXT-046",
        "几何折屏海报",
        "zh",
        "成年汉服女子安静侧身站在几何折屏前，珍珠白服装与奶油白底色相衬。折屏由豆沙粉和雾青色块构成，柔光投下简洁阴影，边缘利落，留白处写“静雅”，现代东方极简海报。",
    ),
    (
        "TXT-047",
        "木屋针织插画",
        "zh",
        "风格化三维动画女孩靠在木屋墙边，穿青红米色拼布针织开衫、深绿T恤和锈色灯芯绒裤，棕色系带靴略有磨损。卷发松挽，抬眼微笑，暖色斜阳与柔和阴影呈现舒适秋日氛围。",
    ),
    (
        "TXT-050",
        "大学几何课堂",
        "zh",
        "真实大学教室中，一位成年数学教师站在黑板旁讲解勾股定理。黑板画直角三角形、三边正方形和清晰公式，讲台放着粉笔与书本，空座位整齐，窗边自然光，手机随拍质感。",
    ),
    (
        "TXT-054",
        "幻彩未来时装",
        "zh",
        "成年东方女性穿带完整内衬的幻彩未来感连衣裙，长发在浅紫、冰蓝和淡粉间渐变。她自然站立，侧颜安静，百褶裙与结构拼接反射珠光，简洁背景，柔光突出发丝和服装材质。",
    ),
    (
        "TXT-057",
        "樱花大头贴",
        "en",
        "A collectible chibi girl and a French bulldog with a pink bow tie pose in a pastel sakura photo booth. A decorated photo strip shows four playful poses with cheerful captions. Warm lights reveal miniature textures, petals and a whimsical arcade setting.",
    ),
]


def normalized_prompt(text):
    return " ".join(unicodedata.normalize("NFKC", text).split()).casefold()


# Short source descriptions are expanded with scene-specific details, never repeated filler.
EXTENSIONS = {
    "TXT-025": "武将的头部略向前倾，面部表情平静，双脚稳定落地，弯刀沿身体外侧向下延伸，避免与长袍重叠而失去轮廓。银色铠甲上的凹槽保留深蓝珐琅，胸甲中央、肩甲边缘和腰带扣具有不同大小的反光面。象牙白织物呈现柔软纤维与重力形成的宽阔褶皱，斗篷翻起的边缘与地面接触部分保持不同明暗。丝带沿头顶上方形成开放弧线，不遮挡脸部与发冠。画面左上方的柔和主光照亮额头、护肩和刀背，右侧较弱的反光保留暗部纹理。人物全身、刀尖和斗篷下缘全部置于方形边界内，四周的灰色留白保持平衡。不要添加其他角色、战场、烟火或说明文字，让人物姿势和材质对比成为视觉中心。",
    "TXT-031": "车身侧面的金属漆呈现连续而柔和的反光，前轮、后轮与底盘的透视一致，轮胎与地面相接的位置有清晰接触阴影。尾翼与后窗之间保留空间，前灯造型完整，玻璃具有较暗的环境反射。画面上方的品牌文字与下方车型名使用统一的无衬线字体，字距均匀，禁止文字覆盖车辆。底部的信息图形用三个简洁模块组织，不填写未经给出的具体性能数值。车头朝右，车尾和尾翼完整落在画面中，四周留出平衡边距。背景的青柠竖条属于海报内部图形，画布本身仍为方形。前景汽车的边缘保持清晰，背景图形稍降低对比，体现视觉主次。避免加入人物、其他汽车、赛道或复杂城市景观。",
    "TXT-033": "透明杯壁的折射使粉色饮料边缘略微弯曲，杯内的冰块和奶泡具有独立层次，杯外零星水珠在侧光下形成细小高光。杯盖与人物接触位置有柔软阴影，双腿自然垂在杯壁外侧，不穿透塑料结构。手提包位于画面另一侧，金色链条沿包身自然弯曲，心形太阳镜放在杯前的路面上，与人物、杯子和包之间保持清楚的尺度关系。镜头略高于地面，但不产生夸张广角变形。柔和树影落在路面，远景只保留低对比的模糊街景。所有品牌图形位于实际产品表面，背景不增加漂浮文字。",
    "TXT-034": "汽水罐位于画面中心稍偏右，罐口朝向左上，金属拉环结构完整，边缘没有变形。罐体前面的水珠大小不一，大水珠呈现放大镜般的局部折射，小水珠形成密集却不均匀的冷凝颗粒。几片柠檬围绕罐体构成开放环形，其中一片切面正对镜头，能够看清放射状果瓣、浅色果皮和柔软的果肉组织。果汁在左右两侧形成弧线，透明薄膜边缘破碎成小水滴，不遮挡汽水罐主要轮廓。两片新鲜绿叶具有可见叶脉，一片位于近景，一片稍远。左侧主灯制造宽阔柔亮的反射，右后方轮廓灯勾勒罐沿和液体边缘，所有高光方向一致。深绿色背景向中央缓慢变亮，不出现地平线和多余道具。画面四周保留空隙，柠檬、叶片和飞溅都不被裁断。金属、果肉、叶片与液体应有可区分的表面质感，保持自然摄影效果，不增加不必要的广告文案。",
    "TXT-036": "笔记本斜放在浅色木桌中央，金属螺旋装订沿左侧延伸，纸张边缘有轻微弯曲和叠层阴影。页面上的横线保持均匀间隔，顶部只写“新的故事”四个清晰字，不出现其他段落。人物从纸面过渡到立体的位置位于腰部附近，铅笔的排线逐渐转化为衣服真实的褶皱，避免生硬断裂。下半身的鞋、裤腿仍由细腻灰色线条勾勒，上半身的卷发与手臂在纸面投下轻微阴影，以强化走出纸页的错觉。人物一只手轻扶页边，另一只手自然向外伸展，姿态放松，眼神看向画面前方。衣服采用浅杏色与灰蓝色，木桌温暖而不过黄。背景只留下几支铅笔和一块橡皮，物件尺寸真实，不能盖住主体。来自左上方的柔和窗光照亮页面和脸部，纸面的亮度低于纯白以保留纤维纹理。整个笔记本和人物都包含在方形画布内，桌面边缘不必出现，整体是有趣而温和的三维错觉插画。",
    "TXT-038": "徽章主体采用盾牌外形，中央是一枚白色足球和一颗蓝色五角星，顶部写“蓝星”，底部写“2026”，文字清晰但不占据太大面积。深蓝珐琅被细金属线分隔，边缘有圆润的镀金包边。徽章表面的高光沿凸起边缘连续延伸，足球凹槽与文字底部保留柔和阴影。布料呈细密斜纹，靠近别针处略微隆起，几根松散纤维自然翘起，但没有大块破损。徽章并非漂浮在布面，底边与面料接触区域具有短而真实的阴影。主光来自左上，右侧用低强度反光板提亮暗部，金属不应出现纯白过曝区域。镜头对准徽章中心，周边布纹逐渐虚化，主体文字与图案都保持可辨。构图稍有倾斜但不歪曲徽章几何形状，方形画面四边保留足够的黑色布料。不要加入第二枚徽章、额外商标或背景人物。",
    "TXT-046": "折屏由三片宽窄略有差别的平面组成，屏风之间的转折形成清楚的几何关系，底部与浅色地面接触，不能看起来悬浮。豆沙粉与雾青色分别占据独立面板，边框细窄，表面哑光，没有雕花。人物位于中下部，衣袖自然垂落，发髻与衣摆完整可见。面部表情安静，身体略向右侧转动，手臂自然放置。光线从左上方柔和照入，屏风后侧在地面留下短而浅的投影，人物与背景保持温和对比。标题只使用“静雅”两字，安排在左上留白区，使用纤细深灰字体，字间距舒展。底部不添加说明段落，画面主要依靠色块、人物轮廓和留白建立节奏。",
    "TXT-047": "人物位于画面左侧稍偏中，手臂自然垂落，双脚稳定落在木屋外的石板小径上。脸颊上的浅淡雀斑与温柔笑容清晰可见，大眼睛有一小块暖色窗光反射。针织开衫的青色、红色、橙色和米色拼布边界保持清楚，每一块织物的针脚方向略有不同，袖口和下摆有自然卷边。灯芯绒裤子的纵向细纹与开衫的粗织纹形成对比，鞋面则带轻微折痕。木墙的纹理沿竖向延伸，几颗钉帽与色差表现自然风化，没有装饰标语。左后方有少量金黄色落叶与一只小花盆，但不能分散主体注意力。斜阳照亮头发、肩膀和衣服边缘，面部暗侧由环境光轻轻补亮。人物全身、鞋子和木墙的一部分都包含在方形画面内，色彩温暖且统一，呈现亲切的动画电影静帧。",
    "TXT-050": "教室采用平视中景构图，教师位于黑板右侧，穿浅蓝衬衫和深灰长裤，右手拿白色粉笔指向直角三角形的斜边，左手自然放在讲台旁。黑板占画面中上方，左侧标题写“勾股定理”，中央画一个直角三角形，短边标a、另一条直角边标b、斜边标c，直角位置用小方框表示。三条边外侧各画一个正方形，分别用白色、浅黄和浅蓝粉笔区分，面积标记与对应边保持清晰关联。右侧只保留公式“a²+b²=c²”，下面给出“3²+4²=5²”的简单例子，排版整齐，没有多余公式。讲台上放着一本打开的教材、粉笔盒和黑板擦，木质表面带有日常使用痕迹。前景保留两排空座位的一部分，桌椅方向一致，不遮挡教师与黑板。窗户位于画面左侧，白天漫射光照亮墙壁和教师的侧脸，黑板表面避免出现覆盖文字的强反射。手机拍摄带轻微自然噪点和真实透视，不使用夸张电影光效。整体呈现正在讲课但没有学生入镜的大学教学环境，黑板上的几何图形和文字应清楚可读。",
}


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def normalize_long(source_id, text):
    text = text.replace("\\n", "\n").strip()
    notes = ["统一方形构图；长版保留主体、场景、材质与光照细节"]
    if source_id == "TXT-002":
        text = text.replace("这张图片中人物五官比例不变，", "一位成年女性，")
        notes.append("去除对未提供参考图的身份保持要求")
    if source_id == "TXT-003":
        text = text.replace("少女", "成年女性").replace("纯欲", "自然")
    if source_id == "TXT-005":
        text = text.replace("8K超高清", "精细清晰的画面")
        notes.append("8K画质口号改为非分辨率描述")
    if source_id == "TXT-007":
        text = (
            text.replace("年轻的中性模特", "成年的中性模特")
            .replace("在他身后", "在她身后")
            .replace("杂志的跨页", "杂志的方形版面")
        )
    if source_id == "TXT-008":
        text = text.replace("vertical 9:16", "square composition")
        text = re.sub(r"[–—-]+ar 9:16.*", "", text)
        notes.append("删除9:16及其他生成工具的ar/stylize/chaos命令")
    if source_id == "TXT-009":
        start = text.index("线条艺术：")
        text = (
            "创作一幅成年女性的红线刺绣肖像，人物长卷发、神态平静，头肩居中。全部肖像细节由绯红色丝线绣在天然米色亚麻布上，不需要参考照片。\n"
            + text[start:]
        )
        text = re.sub(r"对于有胡须的主体，[^。]*。", "", text)
        text = text.replace("纵向 3:4 宽高比。", "方形画幅，肖像与线轴保持完整。")
        notes.append("图像编辑改为独立t2i；具体化人物；删除3:4")
    if source_id == "TXT-017":
        text = text.replace("[Subject]", "Woodland animals: foxes, rabbits, squirrels, owls and deer")
        text = text.replace("9:16 vertical format", "Square format")
        text = text.replace(
            "All characters follow the current theme settings and specified list.",
            "All figures depict foxes, rabbits, squirrels, owls or deer, with distinctive woodland accessories.",
        )
        text = text.replace("without repetition.", "with no identical figure repeated.").replace(
            "non-repeating.", "distinct in pose and accessories."
        )
        notes.append("填充Subject与角色清单；删除竖版比例；完整键盘安排为对角构图")
        text += "\nPlace the complete keyboard diagonally within the square canvas, leaving a small clear margin around every edge."
    if source_id == "MD-01":
        text = text.replace("young Asian woman", "adult Asian woman")
    if source_id == "MD-03":
        text = text.replace("一个身着古风汉服的人物", "一位身着古风汉服的成年女性")
    if source_id == "TXT-025":
        text = text.replace("年轻中国武将", "成年中国武将").replace("垂直 9:16 构图", "1:1方形构图")
    if source_id == "TXT-027":
        text = text.replace("Young female", "Adult woman")
        text = text.replace(
            "wearing the same clothes as in the picture, the clothing design and pattern cannot be changed",
            "wearing a cream linen shirt and blue jeans",
        )
        text = text.replace("cloudy sky", "scattered clouds").replace(
            "random movements", "standing calmly beside the lake"
        )
        notes.append("去除参考图服装依赖，明确服装和静态姿势")
    if source_id == "TXT-028":
        text = text.replace("精致透明织物", "带完整不透明内衬的精细织物")
        text = re.sub(
            r"在海水溅到她身体的地方，.*?非色情。", "海水溅湿裙摆，完整内衬始终保持不透明，外层织物形成自然褶皱。", text
        )
        text = (
            text.replace("半透明的织纹", "细密的织纹")
            .replace("半透明连衣裙", "白色连衣裙")
            .replace("半透明的贴身感", "柔软的垂坠感")
        )
        text = (
            text.replace("2:3的垂直构图", "1:1的方形构图")
            .replace("右侧垂直三分之一处", "右侧三分之一处")
            .replace("2:3", "1:1")
        )
    if source_id == "TXT-029":
        text = (
            text.replace("An ancient-style woman", "An adult woman in pale Hanfu")
            .replace("9:16 vertical composition", "1:1 square composition")
            .replace("9:16", "1:1")
        )
        text = text.replace("appropriate text placed", 'the title "Early Autumn" placed')
    if source_id == "TXT-033":
        text = re.sub(r"关键要求：[^\n]+", "人物为虚构成年女性，深棕短发、温柔笑容，不依赖参考图。", text)
        text = text.replace("（例如草莓拿铁或丝绒白桃）", "（草莓拿铁）")
        notes.append("参考图人脸改为独立虚构成年人物，固定饮品")
    if source_id == "TXT-038":
        text = text.replace("[Team_Name] 2026年FIFA世界杯", "虚构蓝星足球队2026纪念").replace("3:4", "1:1")
        notes.append("填充队名，改为虚构球队纪念徽章")
    if source_id == "TXT-043":
        text = re.sub(
            r"构图・相机：.*?图像处理・完成：",
            "构图・相机：\n1:1方形近距离自拍，脸部位于画面中上部，头顶留有余白，伸出的手臂形成指向脸部的对角线。右侧保留海水，左侧保留弯曲沙滩。保持自然手机广角透视，脸部远离边缘。\n图像处理・完成：",
            text,
            flags=re.S,
        )
        notes.append("重排自拍构图，移除竖幅坐标约束")
    if source_id == "TXT-044":
        text = text.replace("纵向2:3", "1:1方形画幅").replace(
            "裙摆和腿部在画面边缘自然裁切", "裙摆与跪坐的双腿完整包含在画面内"
        )
        text = (
            text.replace("从头顶至膝部附近纳入", "从头顶至跪坐的双脚完整纳入")
            .replace("长银白色头发", "长铂金色头发")
            .replace("铂金金色", "铂金色")
        )
    if source_id == "TXT-046":
        text = text.replace("9:16竖版构图", "1:1方形构图").replace("9:16", "1:1").replace("合适的文字", "标题“静雅”")
    if source_id == "TXT-047":
        text = text.split("2. ", 1)[1]
        notes.append("同段两种变体只选择第二种针织开衫场景")
    if source_id == "TXT-050":
        text = text.replace("XX老师在教XX知识", "一位成年数学教师正在讲解勾股定理")
        notes.append("填充教师与知识点，补充具体课堂内容")
    if source_id == "TXT-054":
        text = text.replace("未来感透视镭射炫彩材质", "带完整内衬的未来感镭射炫彩材质").replace(
            "holographic transparent iridescent material", "fully lined holographic iridescent material"
        )
    if source_id == "TXT-057":
        text = (
            text.replace("foto adjunta miling", "with curled brown hair smiling")
            .replace("Bulldog Francés", "French bulldog")
            .removesuffix(".pmx")
        )
        text += "\nThe booth stands slightly to the left, with its curtain open and a small padded bench visible inside. The girl and puppy sit side by side at matching miniature scale. Keep the printed strip on the right fully visible, with four distinct framed photographs and evenly spaced captions. Soft reflections on the plastic trim contrast with the matte fabric of the bow tie. The background arcade lights remain blurred, while the characters, booth controls and printed strip stay sharp."
        notes.append("去除附图依赖和工具尾缀；四连拍为单图内部布局")
    if source_id in EXTENSIONS:
        text += "\n" + EXTENSIONS[source_id]
        notes.append("补充与主体一致的场景、材质和布局细节，不使用重复填充")
    text = re.sub(r"[48][kK]", "精细", text)
    return text.strip(), notes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=Path(r"D:\Data\pdd\Exp"))
    args = parser.parse_args()
    txt = (args.source_dir / "prompt.txt").read_text(encoding="utf-8-sig")
    md = (args.source_dir / "Hunyuan-Image3.md").read_text(encoding="utf-8-sig")
    source = {f"TXT-{i:03d}": b for i, b in enumerate(re.split(r"\n\s*\n", txt.strip()), 1)}
    source.update(
        {
            f"MD-{i:02d}": b
            for i, b in enumerate(re.findall(r"<summary>Show prompt</summary>\s*(.*?)</details>", md, re.S), 1)
        }
    )
    assert len(source) == 65, "Source segmentation changed; review selections before rebuilding."
    catalog = []
    requests = {"short": [], "long": []}
    for number, (sid, title, language, short) in enumerate(SPECS, 1):
        long, notes = normalize_long(sid, source[sid])
        framing = (
            "采用1:1方形画幅，主体完整置于画面内，四周留出适当空间。"
            if language == "zh"
            else "Use a 1:1 square canvas, keeping the main subject fully inside the frame with balanced margins."
        )
        for group, content in (("short", short), ("long", long)):
            prompt = content + "\n" + framing
            request = dict(
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
            )
            requests[group].append(request)
            catalog.append(
                dict(
                    id=f"scene{number:02d}_{group}",
                    pair_id=f"scene{number:02d}",
                    title=title,
                    group=group,
                    language=language,
                    source_id=sid,
                    source_file="prompt.txt" if sid.startswith("TXT") else "Hunyuan-Image3.md",
                    source_sha256=hashlib.sha256(source[sid].encode()).hexdigest(),
                    prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                    edits=notes + (["提炼同主题短版，保留主体、场景和主要风格"] if group == "short" else []),
                    char_count=len(prompt),
                    user_token_count=None,
                    full_prompt_token_count=None,
                    length_status="relative_pair_not_token_calibrated",
                    request=request,
                )
            )
    assert len({normalized_prompt(r["request"]["prompt"]) for r in catalog}) == 2 * len(SPECS)
    forbidden = r"9[:：]16|3[:：]4|2[:：]3|\b[48]K\b|\[Subject\]|\[Team_Name\]|XX|上传的照片|Figure 1|foto adjunta|same clothes as in the picture|[–—-]+(?:ar|stylize|chaos)\b"
    assert all(not re.search(forbidden, r["request"]["prompt"], re.I) for r in catalog)
    for idx in range(len(SPECS)):
        assert len(requests["long"][idx]["prompt"]) >= 3 * len(requests["short"][idx]["prompt"]), SPECS[idx][0]
    write_jsonl(ROOT / "catalog.jsonl", catalog)
    schedules = {}
    for group in ("short", "long"):
        write_jsonl(ROOT / f"{group}.jsonl", requests[group])
        schedules[f"{group}.jsonl"] = [
            dict(row=i, id=f"scene{i:02d}_{group}", seed=42) for i in range(1, len(SPECS) + 1)
        ]
    mixed = []
    for group in ("short", "long"):
        mixed.extend(zip(requests[group], schedules[f"{group}.jsonl"]))
    # Interleave pairs so every even prefix is exactly half short and half long.
    mixed.sort(key=lambda item: (item[1]["row"], item[1]["id"].endswith("_long")))
    write_jsonl(ROOT / "mixed.jsonl", [r for r, _ in mixed])
    schedules["mixed.jsonl"] = [dict(m, row=i) for i, (_, m) in enumerate(mixed, 1)]
    (ROOT / "schedule_manifest.json").write_text(
        json.dumps(schedules, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    selected = {s[0] for s in SPECS}
    source_groups = {}
    for sid, text in source.items():
        source_groups.setdefault(normalized_prompt(text), []).append(sid)
    audit = dict(
        source_candidate_count=len(source),
        selected_source_count=len(SPECS),
        paired_prompt_count=len(catalog),
        duplicate_source_groups=[ids for ids in source_groups.values() if len(ids) > 1],
        selected_source_ids=sorted(selected),
        excluded_source_ids=sorted(set(source) - selected),
        source_files={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.source_dir.iterdir() if p.is_file()
        },
        note="prompt_utils.py contains prompt-rewriting instructions, not independent t2i samples; it was not executed.",
        length_stats={
            g: dict(
                count=len(requests[g]),
                min_chars=min(len(r["prompt"]) for r in requests[g]),
                max_chars=max(len(r["prompt"]) for r in requests[g]),
            )
            for g in requests
        },
    )
    (ROOT / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    preview = [
        "# Hunyuan PD 实验输入预览",
        "",
        f"{len(SPECS)}个来源主题，每个主题提供同语言长短配对。长度为字符数，尚非模型token数。",
        "",
    ]
    for i in range(len(SPECS)):
        pair = catalog[2 * i : 2 * i + 2]
        preview += [f"## {pair[0]['pair_id']} · {pair[0]['title']} · {pair[0]['source_id']}", ""]
        for row in pair:
            preview += [f"### {row['group']}（{row['char_count']}字符）", "", row["request"]["prompt"], ""]
    (ROOT / "prompts_preview.md").write_text("\n".join(preview), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
