"""
Web聊天服务器 - 育种知识库问答界面
"""
import os
import sys
import json
import threading
import http.server
import socketserver
from urllib.parse import parse_qs

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rag_engine import RAGEngine

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_PATH = os.path.join(BASE_DIR, "templates", "chat.html")

rag_engine = None
chat_history = []


def init_engine():
    global rag_engine
    if rag_engine is None:
        print("初始化RAG引擎...")
        rag_engine = RAGEngine()
        print("RAG引擎就绪!")


class ChatHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path == "/chat":
            init_engine()
            with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
                html = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(html.encode("utf-8"))
        elif self.path == "/api/suggestions":
            suggestions = [
                "Ty1基因和Ty3基因同时携带的材料抗病性会怎样？",
                "糖度大于10的品种怎么评价？",
                "杂交组合选配的原则是什么？",
                "硬度和口感有什么关系？",
                "我的数据里糖度最高的5个品种是哪些？",
                "多抗材料怎么筛选？",
            ]
            self._send_json({"suggestions": suggestions})

    def do_POST(self):
        if self.path == "/api/chat":
            content_length = int(self.headers["Content-Length"])
            body = self.rfile.read(content_length)
            data = json.loads(body)
            question = data.get("question", "")

            if not question.strip():
                self._send_json({"error": "请输入问题"})
                return

            try:
                init_engine()
                result = rag_engine.query(question, chat_history)

                chat_history.append({"role": "user", "content": question})
                chat_history.append({"role": "assistant", "content": result["answer"]})

                self._send_json({
                    "answer": result["answer"],
                    "sources": result["sources"],
                    "intent": result["intent"],
                    "context_count": result["context_count"]
                })
            except Exception as e:
                self._send_json({"error": f"处理失败: {str(e)}"})

    def _send_json(self, data):
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode("utf-8"))

    def log_message(self, format, *args):
        pass


def find_free_port():
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def main():
    port = find_free_port()
    print(f"{'=' * 50}")
    print(f"  番茄育种知识库问答系统")
    print(f"  访问地址: http://localhost:{port}")
    print(f"  按 Ctrl+C 退出")
    print(f"{'=' * 50}")

    with socketserver.TCPServer(("0.0.0.0", port), ChatHandler) as httpd:
        import webbrowser
        threading.Timer(1.0, lambda: webbrowser.open(f"http://localhost:{port}")).start()
        httpd.serve_forever()


if __name__ == "__main__":
    main()
