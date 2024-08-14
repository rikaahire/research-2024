import os
import numpy as np
from gensim.models import Word2Vec
import pickle
import matplotlib.pyplot as plt
import scipy.stats as stats


# Word lists
female_words = ['侄女', '女人', '女儿', '女孩', '女性', '她', '她的', '她自己', '姐妹', '母亲', '阿姨']
male_words = ['他', '他的', '他自己', '侄子', '儿子', '兄弟', '叔叔', '父亲', '男人', '男孩', '男性']
prestige_words = ['享有盛誉', '可敬', '可敬的', '声名显赫', '声誉卓著', '尊敬', '尊贵的', '崇高的', '杰出', '杰出的']
common_words = ['不光彩的', '不名誉的', '卑微的', '声名狼藉的', '平凡的', '普通的', '朴实无华的', '谦虚的', '谦逊的']
education_words = ['受过教育', '受过训练', '受过辅导', '学识渊博', '有文化', '有文采', '知识渊博']
uneducation_words = ['文盲', '无知', '未受过学校教育', '未受过教育', '未受过训练', '未学过', '未经辅导', '没有文采']
affluent_words = ['优势', '优雅的', '偿付能力', '兴旺的', '冲洗', '发达的', '奢侈', '奢侈的', '奢华', '富有', '富裕', 
                    '富裕的', '成功', '无价的', '时髦的', '昂贵的', '更富有', '最富有', '有价值的', '有钱的', '特权的',
                    '珍贵的', '精致', '繁荣的', '豪华', '豪华的', '财产的', '过高的', '高档的']
poor_words = ['一文不值', '一贫如洗', '不发达', '不成功', '不繁荣', '乞丐', '吝啬', '基本', '平原', '廉价', '弱势',
                '必需品', '摇摇欲坠', '无力', '无钱', '最穷', '毫无价值', '没钱的', '皮肤', '破产', '破烂', '经济',
                '肮脏', '贫困', '贫困的', '贫穷', '身无分文的', '较穷']
good_words = ['不错', '令人愉快', '令人钦佩', '值得信赖', '值得称赞', '启发', '好', '富有同情心', '幸福', '建设性',
                '忠诚', '慷慨', '有德', '有益', '欢迎', '正派', '正直', '真诚', '积极', '诚实', '贞洁', '道德']
bad_words = ['下流的', '不可靠的', '不诚实的', '不道德的', '令人不快的', '可怕的', '可鄙的', '吝啬的', '坏的',
                '应受谴责的', '敌对的', '有害的', '有罪的', '欺骗的', '残忍的', '消极的', '破坏性的', '背叛的',
                '虚伪的', '诅咒的', '误导的', '邪恶的']
strong_words = ['主权', '占主导地位', '坚固', '大', '大胆', '强健', '强大', '指挥', '控制', '无所不能的', '有弹性',
                '有影响力', '有能力', '权威', '自信', '至高无上的', '顽强']
weak_words = ['依赖', '外围', '娇弱', '小', '屈从', '屈服', '无关紧要', '无效', '无能', '温顺', '琐碎', '胆小', '脆弱',
                '虚弱', '顺从']
active_words = ['快速', '活泼', '响亮', '尖锐', '活跃']
passive_words = ['慢', '安静', '无声', '静态', '平静']


# Jobs
jobs = [
    "会计师", "演员", "女演员", "保险精算师", "顾问", "助手", "大使", "动画师",
    "弓箭手", "艺术家", "宇航员", "天文学家", "运动员", "律师", "拍卖师", "作家",
    "保姆", "面包师", "芭蕾舞演员", "银行家", "理发师", "棒球运动员", "篮球运动员",
    "行李员", "生物学家", "铁匠", "簿记员", "投球手", "建筑工人", "屠夫", "管家",
    "出租车司机", "书法家", "船长", "心脏病专家", "护理员", "木匠", "制图师",
    "漫画家", "收银员", "捕手", "餐饮服务商", "大提琴手", "牧师", "司机", "厨师",
    "化学家", "办事员", "教练", "鞋匠", "作曲家", "门房",
    "领事", "承包商", "警察", "验尸官", "信使", "密码学家", "保管员", "舞者", "牙医",
    "副手", "皮肤科医生", "设计师", "侦探", "独裁者", "导演", "唱片骑师", "潜水员",
    "医生", "门卫", "鼓手", "干洗店", "生态学家", "经济学家", "编辑", "教育家", "电工",
    "皇帝", "皇后", "工程师", "艺人", "昆虫学家", "企业家", "高管", "探险家", "出口商",
    "灭虫者", "（电影中的）临时演员", "猎鹰人", "农民", "金融家", "消防员", "渔夫",
    "长笛演奏家", "足球运动员", "工头", "游戏设计师", "垃圾工", "园丁", "采集者",
    "宝石切割工", "将军", "遗传学家", "地理学家", "地质学家", "高尔夫球手", "州长",
    "杂货商", "导游", "杂工", "竖琴师", "公路巡警", "流浪汉", "猎人", "插画家", "进口商",
    "讲师", "实习生", "内科医生", "翻译", "发明家", "调查员", "狱吏", "看门人", "小丑",
    "珠宝商", "骑师", "记者", "法官", "空手道老师", "劳工", "房东", "园艺师", "洗衣女工",
    "律师", "讲师", "法律助理", "图书管理员", "剧本作家", "救生员", "语言学家", "说客",
    "锁匠", "作词家", "魔术师", "女仆", "邮递员", "经理", "制造商", "海军", "营销人员",
    "泥瓦匠", "数学家", "市长", "机械师", "信使", "助产士", "矿工", "模特", "僧侣", "壁画家",
    "音乐家", "航海家", "谈判者", "公证人", "小说家", "修女", "护士", "双簧管演奏者", "操作员",
    "眼科医生", "配镜师", "神谕者", "勤务员", "鸟类学家", "画家", "古生物学家", "律师助理",
    "公园管理员", "病理学家", "当铺老板", "小贩", "儿科医生", "打击乐手", "表演者", "药剂师",
    "慈善家", "哲学家", "摄影师", "物理学家", "钢琴家", "飞行员", "投手", "水管工", "诗人",
    "男警察", "女警察", "政客", "总统", "王子", "公主", "校长", "私人", "私家侦探", "制片人",
    "教授", "程序员", "精神病学家", "心理学家", "出版商", "四分卫", "缝纫工", "放射科医生",
    "牧场主", "护林员", "房地产经纪人", "接待员", "裁判", "登记员", "记者", "代表", "研究员",
    "餐馆老板", "零售商", "退休人员", "水手", "销售员", "武士", "萨克斯管演奏家", "学者",
    "科学家", "侦察员", "潜水员", "裁缝", "保安警卫", "参议员", "警长", "歌手", "铁匠",
    "社交名流", "士兵", "间谍", "明星", "统计员", "股票经纪人", "街道清洁工", "学生",
    "外科医生", "测量员", "游泳者", "税务员", "动物标本剥制师", "教师", "技术员",
    "网球运动员", "试飞员", "铺瓦工", "工具制造者", "交易员", "培训师", "垃圾收集者",
    "旅行社", "财务主管", "卡车司机", "导师", "打字员", "裁判员", "殡葬工", "引座员", "男仆",
    "退伍军人", "兽医", "牧师", "小提琴手", "服务员", "女服务员", "监狱长", "战士", "钟表匠",
    "织工", "焊工", "木雕师", "牧马人", "作家", "木琴演奏家", "约德尔歌手", "动物园饲养员",
    "动物学家"
]

# Path to the models directory
models = '/scratch/network/sa3937/wordembed/w2v_dec_nopunc/models_dec'
        
# Function to calculate distances
def calculate_dist(model1, model2, words, jobs):
    min_dist_1 = []
    min_dist_2 = []
    for job in jobs:
        dist_1 = []
        dist_2 = []
        for word in words:
            if word in model1.wv and job in model1.wv and word in model2.wv and job in model2.wv:
                d1 = abs(model1.wv.similarity(word, job))
                dist_1.append(d1)
                d2 = abs(model2.wv.similarity(word, job))
                dist_2.append(d2)
        if dist_1 and dist_2:
            min_dist_1.append(min(dist_1))
            min_dist_2.append(min(dist_2))
    return (min_dist_1, min_dist_2) if min_dist_1 and min_dist_2 else None

# Load models
for file in os.listdir(models):
    if file.endswith('1970_model.pkl'):
        model_file = os.path.join(models, file)
        with open(model_file, 'rb') as f:
            model1 = pickle.load(f)
            
    if file.endswith('2000_model.pkl'):
        model_file = os.path.join(models, file)
        with open(model_file, 'rb') as f:
            model2 = pickle.load(f)

# Dictionary to store results
results = {}

# Word pairs
word_pairs = [
    (female_words, male_words, 'Female vs Male'),
    (prestige_words, common_words, 'Prestige vs Common'),
    (education_words, uneducation_words, 'Education vs Uneducation'),
    (affluent_words, poor_words, 'Affluent vs Poor'),
    (good_words, bad_words, 'Good vs Bad'),
    (strong_words, weak_words, 'Strong vs Weak'),
    (active_words, passive_words, 'Active vs Passive')
]

# Open output file
with open('/scratch/network/sa3937/wordembed/w2v_dec_nopunc/correlation_rank.txt', 'w') as f_out:
    # Iterate over each word pair
    for word_list_1, word_list_2, label in word_pairs:
        # Calculate distances for 1st and 2nd word lists
        result_1 = calculate_dist(model1, model2, word_list_1, jobs)
        result_2 = calculate_dist(model1, model2, word_list_2, jobs)
        
        if result_1 is not None and result_2 is not None:
            min_dist_1_1, min_dist_2_1 = result_1
            min_dist_1_2, min_dist_2_2 = result_2
            
            # Difference in distances for model (decade) 1
            diff_dist_1 = np.array(min_dist_1_1) - np.array(min_dist_1_2)
            
            # Difference in distances for model (decade) 2
            diff_dist_2 = np.array(min_dist_2_1) - np.array(min_dist_2_2)
            
            # Rank occupations based on differences for 1st decade
            ranks_dec_1 = np.argsort(diff_dist_1)
            
            # Get rank-based correlation
            corr_dec_1, p_value_dec_1 = stats.spearmanr(ranks_dec_1, diff_dist_1)
            corr_dec_2, p_value_dec_2 = stats.spearmanr(ranks_dec_1, diff_dist_2)
            
            corr_dec_1_ken, p_value_dec_1_ken = stats.kendalltau(ranks_dec_1, diff_dist_1)
            corr_dec_2_ken, p_value_dec_2_ken = stats.kendalltau(ranks_dec_1, diff_dist_2)
            
            
            # Save results to dictionary
            results[label] = {
                'Spearman Dec 1': corr_dec_1,
                'Spearman Dec 2': corr_dec_2,
                'Kendall Dec 1': corr_dec_1_ken,
                'Kendall Dec 2': corr_dec_2_ken
            }
            
            # Write results to file
            f_out.write(f"{label}:\n")
            f_out.write(f"Spearman correlation coefficient for dec 1: {corr_dec_1}\n")
            f_out.write(f"Spearman p-value for dec 1: {p_value_dec_1}\n")
            f_out.write(f"Spearman correlation coefficient for dec 2: {corr_dec_2}\n")
            f_out.write(f"Spearman p-value for dec 2: {p_value_dec_2}\n")
            f_out.write(f"Kendall tau for dec 1: {corr_dec_1_ken}\n")
            f_out.write(f"Kendall tau p-value for dec 1: {p_value_dec_1_ken}\n")
            f_out.write(f"Kendall tau for dec 2: {corr_dec_2_ken}\n")
            f_out.write(f"Kendall tau p-value for dec 2: {p_value_dec_2_ken}\n\n")
            
            print(f"{label}:")
            print("Spearman correlation coefficient for dec 1:", corr_dec_1)
            print("Spearman p-value for dec 1:", p_value_dec_1)
            print("Spearman correlation coefficient for dec 2:", corr_dec_2)
            print("Spearman p-value for dec 2:", p_value_dec_2)
            print("Kendall tau for dec 1:", corr_dec_1_ken)
            print("Kendall tau p-value for dec 1:", p_value_dec_1_ken)
            print("Kendall tau for dec 2:", corr_dec_2_ken)
            print("Kendall tau p-value for dec 2:", p_value_dec_2_ken)
            print()
            