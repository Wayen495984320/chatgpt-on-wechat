from bot.bot import Bot
from bot.session_manager import SessionManager
from common.log import logger
from common.token_bucket import TokenBucket
from config import conf
from openai import OpenAI
import time
from .deepseek_session import DeepseekSession
from common import const

class DeepseekBot(Bot):
    def __init__(self):
        super().__init__()
        self.token_bucket = TokenBucket(conf().get("rate_limit_deepseek", 20))
        # 初始化 OpenAI 客户端
        self.client = OpenAI(
            api_key=conf().get("deepseek_api_key"),
            base_url="https://api.deepseek.com/v1"
        )
        self.sessions = SessionManager(DeepseekSession, model=const.DEEPSEEK_REASONER)

    def reply(self, query, context=None):
        """
        调用 DeepSeek API 回复消息
        :param query: 用户输入的消息
        :param context: 上下文信息
        :return: 回复内容
        """
        if not self.token_bucket.consume(1):
            return "请求太快啦，请稍后重试"

        try:
            session = self.sessions.session_query(query, context.get("session_id"))
            reply_content = self.reply_text(session, retry_count=0)
            
            if reply_content["completion_tokens"] == 0 and len(reply_content["content"]) > 0:
                return reply_content["content"]
            elif reply_content["completion_tokens"] > 0:
                self.sessions.session_reply(reply_content["content"], session.session_id, reply_content["total_tokens"])
                return reply_content["content"]
            else:
                return "抱歉，我现在无法回答"

        except Exception as e:
            logger.exception(e)
            return f"DeepSeek API 异常：{str(e)}"

    def reply_text(self, session, retry_count=0) -> dict:
        """
        调用 DeepSeek API 获取回复
        :param session: 会话信息
        :param retry_count: 重试次数
        :return: dict
        """
        try:
            if not self.token_bucket.get_token():
                raise Exception("请求太快啦，请休息一下再问我吧")

            # 创建聊天完成请求
            completion = self.client.chat.completions.create(
                model=const.DEEPSEEK_REASONER,
                messages=session.messages,
                stream=True
            )

            # 处理流式响应
            full_content = ""
            completion_tokens = 0
            total_tokens = 0

            for chunk in completion:
                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta
                if not delta:
                    continue
                    
                # 获取回复内容
                if hasattr(delta, 'content') and delta.content:
                    full_content += delta.content
                
                # 获取 token 统计
                if hasattr(chunk, 'usage'):
                    completion_tokens = chunk.usage.completion_tokens
                    total_tokens = chunk.usage.total_tokens

            return {
                "completion_tokens": completion_tokens or len(full_content),
                "total_tokens": total_tokens or len(full_content),
                "content": full_content
            }

        except Exception as e:
            need_retry = retry_count < 2
            error_msg = "我现在有点累了，等会再来吧"
            
            if "rate limit" in str(e).lower():
                error_msg = "提问太快啦，请休息一下再问我吧"
                if need_retry:
                    time.sleep(20)
            elif "timeout" in str(e).lower():
                error_msg = "我没有收到你的消息"
                if need_retry:
                    time.sleep(5)
            else:
                logger.exception(f"DeepSeek API 异常: {e}")
                need_retry = False
                self.sessions.clear_session(session.session_id)

            if need_retry:
                logger.warning(f"第{retry_count + 1}次重试")
                return self.reply_text(session, retry_count + 1)
            
            return {
                "completion_tokens": 0,
                "total_tokens": 0,
                "content": error_msg
            }
