from openai import OpenAI
from threading import Condition, Thread, Lock
import os
import json
import difflib
import re
from ui import console


class Retrieve:
    def __init__(self, api_key, base_url, category: str = "papert"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)

        self.category = category or "papert"

        self.running = False
        self.retrieve_thread = None

        self.type_files = []
        self.types = []

        self._result_lock = Lock()
        self.have_new_result = False
        self.result = ""

        self._input_cond = Condition()
        self.have_new_input = False
        self.input = ""

    def start(self):
        self.running = True
        self.retrieve_thread = Thread(target=self.spin, daemon=True)
        self.retrieve_thread.start()

    def spin(self):
        while True:
            with self._input_cond:
                while self.running and not self.have_new_input:
                    self._input_cond.wait()
                if not self.running:
                    return
                current_input = self.input
                self.have_new_input = False

            console.print("[dim]开始检索...[/]")
            type_name = self._retrieve(current_input)
            with self._result_lock:
                self.result = type_name
                self.have_new_result = True

    def stop(self):
        with self._input_cond:
            self.running = False
            self._input_cond.notify_all()
        if self.retrieve_thread is not None:
            self.retrieve_thread.join(timeout=2.0)
            self.retrieve_thread = None

    def load_type_library(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        base_type_library_path = os.path.join(current_dir, "../TypeLibrary")
        type_library_path = os.path.join(base_type_library_path, self.category)

        if not os.path.isdir(type_library_path):
            console.print(
                f"[yellow]检索: 目录 {type_library_path} 不存在，改用 {base_type_library_path}[/]"
            )
            type_library_path = base_type_library_path

        json_path = os.path.join(type_library_path, "_type_info.json")

        loaded_from_json = False
        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    self.types = data
                    self.type_files = [t.get("id") for t in self.types if "id" in t]
                    loaded_from_json = True
            except Exception as e:
                console.print(f"[red]读取 _type_info.json 失败: {e}[/]")

        if not loaded_from_json:
            self.type_files = [
                f[:-4] for f in os.listdir(type_library_path) if f.endswith(".txt")
            ]
            self.types = [{"id": name} for name in self.type_files]
        console.print(
            f"[green]已加载手势列表[/] (类别={self.category}): {self.type_files}"
        )

    def _local_score(self, query: str, gesture: dict) -> float:
        q = query.lower().strip()
        gid = gesture.get("id", "") or ""
        name = gesture.get("name", "") or ""
        usage = gesture.get("usage", "") or ""
        intents = gesture.get("intents", []) or []

        base = difflib.SequenceMatcher(None, q, gid).ratio()

        bonus_name = 0.15 if name and name in q else 0.0

        bonus_intent = 0.0
        for it in intents:
            if isinstance(it, str) and it and it.lower() in q:
                bonus_intent += 0.1
        bonus_intent = min(bonus_intent, 0.3)

        tokens = re.findall(r"[a-zA-Z]+", usage.lower())
        token_hit = sum(1 for t in tokens if len(t) > 3 and t in q)
        bonus_usage = min(token_hit * 0.05, 0.15)

        score = base + bonus_name + bonus_intent + bonus_usage
        return score

    def _local_retrieve(self, query: str):
        best = None
        best_score = 0.0
        for g in self.types:
            s = self._local_score(query, g)
            if s > best_score:
                best_score = s
                best = g.get("id")
        return best, best_score

    def _retrieve(self, query: str):
        if not self.type_files:
            return None

        local_id, score = self._local_retrieve(query)
        if local_id and score >= 0.75:
            return local_id

        brief_lines = []
        for t in self.types:
            intents = (
                ",".join(t.get("intents", [])[:3])
                if isinstance(t.get("intents"), list)
                else ""
            )
            brief_lines.append(f"{t.get('id')}: {t.get('pose', '')}; intents={intents}")
        catalog = "\n".join(brief_lines)

        prompt = (
            "You are a gesture type selector. Given a natural language user query (maybe noisy ASR). "
            "Choose the best gesture id from the catalog. If nothing fits, answer None. Just output the id or None.\n"
            f"Catalog:\n{catalog}\nQuery: {query}\nAnswer:"
        )

        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a concise classifier returning only a gesture id or None.",
                    },
                    {"role": "user", "content": prompt},
                ],
                stream=False,
            )
            candidate = response.choices[0].message.content.strip()
        except Exception as e:
            console.print(f"[red]大模型检索出错: {e}[/]")
            candidate = None

        if candidate in self.type_files:
            return candidate

        if local_id and score >= 0.55:
            return local_id
        return None

    def retrieve(self, query: str):
        with self._input_cond:
            self.input = query
            self.have_new_input = True
            self._input_cond.notify_all()

    def has_new_result(self):
        with self._result_lock:
            return self.have_new_result

    def get(self):
        try:
            with self._result_lock:
                if self.have_new_result:
                    self.have_new_result = False
                    return self.result
                else:
                    return None
        except Exception as e:
            console.print(f"[red]获取检索结果出错: {e}[/]")
            return None
