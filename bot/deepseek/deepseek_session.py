from bot.session_manager import Session
from common.log import logger
from common import const

class DeepseekSession(Session):
    def __init__(self, session_id, system_prompt=None, model=const.DEEPSEEK_REASONER):
        super().__init__(session_id, system_prompt)
        self.model = model
        self.reset()

    def reset(self):
        """
        重置会话
        """
        system_item = {
            "role": "system",
            "content": self.system_prompt if self.system_prompt else "You are a helpful assistant."
        }
        self.messages = [system_item]
        return self

    def discard_exceeding(self, max_tokens, cur_tokens=None):
        """
        丢弃超出 max_tokens 的消息
        """
        precise = True
        try:
            cur_tokens = self.calc_tokens()
        except Exception as e:
            precise = False
            if cur_tokens is None:
                raise e
            logger.debug("Exception when counting tokens precisely for query: {}".format(e))

        while cur_tokens > max_tokens:
            if len(self.messages) > 2:
                self.messages.pop(1)
            elif len(self.messages) == 2 and self.messages[1]["role"] == "assistant":
                self.messages.pop(1)
                if precise:
                    cur_tokens = self.calc_tokens()
                else:
                    cur_tokens = cur_tokens - max_tokens
                break
            elif len(self.messages) == 2 and self.messages[1]["role"] == "user":
                logger.warn("user message exceed max_tokens. total_tokens={}".format(cur_tokens))
                break
            else:
                logger.debug("max_tokens={}, total_tokens={}, len(messages)={}".format(max_tokens, cur_tokens, len(self.messages)))
                break
            if precise:
                cur_tokens = self.calc_tokens()
            else:
                cur_tokens = cur_tokens - max_tokens
        return cur_tokens

    def calc_tokens(self):
        """
        计算当前消息使用的 token 数
        """
        return num_tokens_from_messages(self.messages, self.model)

def num_tokens_from_messages(messages, model):
    """
    计算消息列表使用的 token 数
    """
    if model in [const.DEEPSEEK_CHAT, const.DEEPSEEK_REASONER]:
        # DeepSeek 模型暂时使用字符数来估算 token 数
        # 可以根据实际情况调整计算方法
        return num_tokens_by_character(messages)

    # 如果不是 DeepSeek 模型，使用原有的计算方法
    import tiktoken
    try:
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")  # 使用通用编码
        tokens_per_message = 4
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
        num_tokens += 3  # 每个回复都以 <|start|>assistant<|message|> 开头
        return num_tokens
    except Exception as e:
        logger.debug(f"Token counting error for model {model}, using character count: {e}")
        return num_tokens_by_character(messages)

def num_tokens_by_character(messages):
    """
    使用字符数估算 token 数
    """
    tokens = 0
    for msg in messages:
        tokens += len(msg["content"])
    return tokens
