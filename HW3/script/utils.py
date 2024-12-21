from transformers import BitsAndBytesConfig
import torch


def get_prompt(instruction: str) -> str:
    '''Format the instruction as a prompt for LLM.'''
    # return f"你是人工智慧助理，以下是用戶和人工智能助理之間的對話。你要對用戶的問題提供有用、安全、詳細和禮貌的回答。USER: {instruction} ASSISTANT:"
    return f"你是一位精通白話文與文言文的大師，以下是用戶與你之間的對話，你的目標是對用戶提供有用、精確且簡潔的回答。USER:{instruction}ASSISTANT:"

def get_prompt_zero_shot(instruction: str) -> str : 
    ''' Format the instruction as a prompt for LLM.'''
    return f"你是專業古文學者，以下是用戶和專業古文者之間的對話。你要對用戶的問題提供精確、安全、詳細的回答。用戶:{instruction} 專業古文學者:"

def get_prompt_few_shot(instruction: str) -> str :
    """ Format the instruction as a prompt for LLM."""
    prompt = (
        f"你是專業古文學者，以下是用戶和專業古文者之間的對話。你要對用戶的問題提供精確、安全、詳細的回答。"
        f"用戶: 幫我進行以下句子的文言文翻譯：契丹主以陽城之戰為彥卿所敗，詰之。彥卿曰： 臣當時惟知為晉主竭力，今日死生惟命。 "
        f"專業古文學者:契丹主因陽城之戰被符彥卿打敗，追問符彥卿，彥卿說： 臣當時隻知為晉主竭盡全力，今日死生聽你決定。 "
        f"用戶: 將下面句子翻譯成文言文：等脩行師到達，腹背攻擊他，脩行師大敗，因而乞求投降，陸子隆同意他投降，將他送於京師。 "
        f"專業古文學者:及行師至，腹背擊之，行師大敗，因乞降，子隆許之，送於京師。 "
        f"用戶: 臧霸自亡匿，操募索得之，使霸招吳敦、尹禮、孔觀等，皆詣操降。幫我把這句話翻譯成現代文 "
        f"專業古文學者:臧霸自己逃到民間隱藏起來，曹操懸賞將他捉拿，派他去招降吳敦、尹禮、孫觀等，這些人全都到曹操營中歸降。 "
        f"用戶: 因為郊祀禮重大，不宜由臣子代行，請求等到聖上身體康復，改換占蔔吉日行禮。這句話在中國古代怎麼說 "
        f"專業古文學者:蓋以郊祀禮重，不宜攝以人臣，請俟聖躬痊，改蔔吉日行禮。 "
        f"用戶: 翻譯成文言文：於是，廢帝讓瀋慶之的堂侄、直將軍瀋攸之賜瀋慶之毒藥，命瀋慶之自殺。 "
        f"專業古文學者:帝乃使慶之從父兄子直閣將軍攸之賜慶之藥。 "
        f"用戶: {instruction} 專業古文學者:"
    )
    return prompt 

def get_bnb_config() -> BitsAndBytesConfig:
    '''Get the BitsAndBytesConfig for 4-bit quantization.'''
    return BitsAndBytesConfig(
        load_in_4bit = True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

