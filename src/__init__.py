from rag import SimpleRAG
from pathlib import Path


def main():
    rag = SimpleRAG()
    # 构建索引，默认从项目根目录下的 data 文件夹读取文档
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    # 构建索引可能需要一些时间，取决于文档数量和大小
    rag.build_index(str(data_dir),force_rebuild=False)
    
    # 进入交互式问答环节
    while True:
        query = input("\n请输入问题，输入 exit 退出：")

        if query.lower() in ["exit", "quit"]:
            break

        print("\n回答：")
        for token in rag.ask_stream(query):
            print(token, end="", flush=True)
        print()


if __name__ == "__main__":
    main()
