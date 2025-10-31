import arxiv
from config import DEFAULT_MAX_RESULTS
                    
class ArxivPaperSearcher:
    def __init__(self, query, max_results_per_batch=DEFAULT_MAX_RESULTS, sort_by=arxiv.SortCriterion.SubmittedDate):
        if max_results_per_batch > 300:
            raise ValueError("arXiv 单次最多返回 300 篇论文")
        self.query = query
        self.max_results_per_batch = max_results_per_batch
        self.all_papers_list = []
        self.all_papers_list_dict = []
        self.processed_papers = set() # 初始化空集合
        self.current_start = 0  # 在“时间范围结果集”内的偏移

    def get_next_batch(self):
        batch_papers_list = []
        batch_papers_list_dict = []

        client = arxiv.Client(
            page_size=self.max_results_per_batch,
            delay_seconds=3,
            num_retries=3
        )

        search = arxiv.Search(query=self.query, sort_by=arxiv.SortCriterion.SubmittedDate)
        results = client.results(search, offset=self.current_start)

        count = 0
        try:
            for result in results:
                if count >= self.max_results_per_batch:
                    break
                if result.entry_id in self.processed_papers:
                    continue

                paper_text = f"Title: {result.title}\nSummary: {result.summary}"
                batch_papers_list.append(paper_text)
                batch_papers_list_dict.append({
                    'title': result.title,
                    'summary': result.summary,
                    'entry_id': result.entry_id
                })
                self.processed_papers.add(result.entry_id)
                count += 1

        except arxiv.UnexpectedEmptyPageError:
            print("⚠️  已无更多论文（在当前检索范围内）")
        except Exception as e:
            print(f"❌ 获取论文时出错: {e}")

        self.current_start += self.max_results_per_batch
        self.all_papers_list.extend(batch_papers_list)
        self.all_papers_list_dict.extend(batch_papers_list_dict)
        return batch_papers_list, batch_papers_list_dict

    def get_total_count(self):
        return len(self.all_papers_list)