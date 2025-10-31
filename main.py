from datetime import datetime, timedelta
from config import (DEFAULT_TOPK,
                    DEFAULT_THRESHOLD,
                    DEFAULT_QUERY,
                    DEFAULT_DAYS_TO_SEARCH,
                    DEFAULT_TARGET_TITLE,
                    DEFAULT_TARGET_SUMMARY)
from arxiv_fetch import ArxivPaperSearcher
from computer_embedding import get_paper_by_embedding, initialize_model

def main():
    
    # 构建带时间范围的查询
    end_date = datetime.utcnow().strftime('%Y%m%d')
    start_date = (datetime.utcnow() - timedelta(days=DEFAULT_DAYS_TO_SEARCH)).strftime('%Y%m%d')
    query = f"{DEFAULT_QUERY} AND submittedDate:[{start_date} TO {end_date}]"

    print("=" * 60)
    print(f"🚀 开始检索 arXiv 上最近 {DEFAULT_DAYS_TO_SEARCH} 天的论文...")
    print(f"🔍 查询语句: {query}")
    print("=" * 60)

    searcher = ArxivPaperSearcher(query=query)

    # 加载模型（提前加载，避免后面卡顿）
    print("\n🧠 正在加载语义模型（首次运行会下载）...")
    model = initialize_model()
    print("✅ 模型加载完成。\n")

    print("请选择你的模式：\n")
    print("1. 分批次获取")
    print("2. 一次性获取")
    decision =  input("\n请做出选择 (1/2): ").strip()
    if decision == '1': # 注意不是1而是'1'
        # 交互主循环
        while True:
            total = searcher.get_total_count()
            next_start = searcher.current_start
            next_end = next_start + searcher.max_results_per_batch

            print(f"\n📊 当前已缓存论文数量: {total}")
            print(f"📥 下一批将获取论文索引范围: [{next_start} ~ {next_end})")
            print("\n请选择操作：")
            print("1. 获取下一批论文")
            print("2. 基于当前所有论文进行相似度搜索")
            print("3. 退出")

            choice = input("\n请输入选项 (1/2/3): ").strip() # strip把无关字符滤掉，只关注选项数字

            if choice == '1':
                print(f"\n⏳ 正在获取第 [{next_start} ~ {next_end}) 篇论文...")
                batch_papers, _ = searcher.get_next_batch()

                if not batch_papers:
                    print("📌 本次未获取到新论文（可能已到末尾）")
                else:
                    print(f"✅ 成功获取 {len(batch_papers)} 篇新论文！")
                    print(f"📚 当前总论文数: {searcher.get_total_count()}")

            elif choice == '2':
                if searcher.get_total_count() == 0:
                    print("❗ 请先获取至少一批论文再进行相似度搜索。")
                    continue

                print(f"\n🔍 正在对 {searcher.get_total_count()} 篇论文进行语义匹配...")
                results = get_paper_by_embedding(model = model, all_papers_list = searcher.all_papers_list, target_title = DEFAULT_TARGET_TITLE, target_summary = DEFAULT_TARGET_SUMMARY)

                if not results:
                    print(f"❌ 未找到相似度 ≥ {DEFAULT_THRESHOLD} 的论文。")
                else:
                    print(f"\n✅ 找到 {len(results)} 篇高度相似(相似度大于{DEFAULT_THRESHOLD})的论文（Top-{DEFAULT_TOPK}）：\n")
                    for rank, (idx, sim) in enumerate(results, 1): # 加上1，从1开始编号
                        paper = searcher.all_papers_list_dict[idx]
                        print(f"【{rank}】相似度: {sim:.4f}")
                        print(f"标题: {paper['title']}")
                        print(f"摘要: {paper['summary']}")
                        print("-" * 80)

            elif choice == '3':
                print("👋 程序已退出，感谢使用！")
                break

            else:
                print("❗ 无效输入，请输入 1、2 或 3。")
    elif decision == '2':
            # 自动获取所有论文（最多 10,000 篇）
        print("\n⏳ 正在自动下载论文（请耐心等待，可能需要几分钟）...\n")
        current_count = 0
        current_count_repeat = 0
        while True:
            prev_count = searcher.get_total_count()
            batch = searcher.get_next_batch()
            last_current_count = current_count
            current_count = searcher.get_total_count()
            if not batch:
                break
            print(f"✅ 已获取 {current_count} 篇论文...")
            if current_count == last_current_count:
                current_count_repeat += 1
                if current_count_repeat > 5:
                    break
            else:
                current_count_repeat = 0

        total = searcher.get_total_count()
        if total == 0:
            print("❌ 未找到任何论文，请检查网络或调整时间范围。")
            return

        print(f"\n🎉 成功获取 {total} 篇论文！\n")

        # 相似度搜索
        print("\n🔍 正在进行相似度匹配...")
        results = get_paper_by_embedding(model, searcher.all_papers_list, target_title = DEFAULT_TARGET_TITLE, target_summary = DEFAULT_TARGET_SUMMARY)
        # 输出结果
        if not results:
            print(f"❌ 未找到相似度 ≥ {DEFAULT_THRESHOLD} 的论文。")
        else:
            print(f"\n✅ 找到 {len(results)} 篇高度相似的论文（Top-{DEFAULT_TOPK}）：\n")
            for rank, (idx, sim) in enumerate(results, 1):
                paper = searcher.all_papers_list_dict[idx]
                print(f"【{rank}】相似度: {sim:.4f}")
                print(f"标题: {paper['title']}")
                print(f"摘要: {paper['summary']}")
                print("-" * 80)

        print("\n🔚 程序运行完毕！")
    else:
        print("❗ 无效输入，请输入 1 或 2。")


if __name__ == "__main__":
    main()