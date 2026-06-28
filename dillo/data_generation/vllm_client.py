"""OpenAI-compatible client for local vLLM servers."""

from __future__ import annotations

import base64
import re
import time
from typing import Iterable, Optional

import cv2
from openai import OpenAI


class VLLMClient:
    """Small wrapper around a vLLM OpenAI-compatible chat endpoint."""

    def __init__(
        self,
        model: str,
        host: str = "127.0.0.1",
        port: int = 8000,
        api_key: str = "password",
        system_prompt: str = "",
        max_retries: int = 3,
        retry_delay: float = 5.0,
        enable_thinking: bool = True,
        max_tokens: int = 4096,
        temperature: float = 0.2,
    ):
        self.model = model
        self.system_prompt = system_prompt
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.enable_thinking = enable_thinking
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.client = OpenAI(base_url=f"http://{host}:{port}/v1", api_key=api_key)

    @staticmethod
    def _image_content(frame) -> dict:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        ok, buffer = cv2.imencode(".jpg", frame_bgr)
        if not ok:
            raise ValueError("Could not JPEG-encode frame")
        b64 = base64.b64encode(buffer).decode("utf-8")
        return {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"},
        }

    def __call__(
        self,
        message_query: str,
        frames: Optional[Iterable] = None,
        timestamped_frames: Optional[Iterable[tuple[float, object]]] = None,
        return_thinking: bool = False,
    ):
        content = [{"type": "text", "text": message_query}]
        if timestamped_frames is not None:
            for ts_sec, frame in timestamped_frames:
                content.append({"type": "text", "text": f"<{ts_sec:.2f} seconds>"})
                content.append(self._image_content(frame))
        elif frames is not None:
            for frame in frames:
                content.append(self._image_content(frame))

        call_kwargs = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": content},
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "extra_body": {
                "chat_template_kwargs": {"enable_thinking": self.enable_thinking}
            },
        }

        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                completion = self.client.chat.completions.create(**call_kwargs)
                raw = completion.choices[0].message.content or ""
                thinking = ""
                clean = raw.strip()
                if "</think>" in raw:
                    blocks = re.findall(r"<think>(.*?)</think>", raw, flags=re.DOTALL)
                    thinking = "\n".join(block.strip() for block in blocks)
                    clean = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
                return (clean, thinking) if return_thinking else clean
            except Exception as exc:
                last_error = exc
                print(f"[vLLM] attempt {attempt}/{self.max_retries} failed: {exc}")
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay)

        error_msg = f"[VLLM_ERROR] {last_error}"
        return (error_msg, "") if return_thinking else error_msg
