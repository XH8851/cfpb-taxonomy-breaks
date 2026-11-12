"""Build paper/references.bib for T2 (authors/titles/years never typed by hand for arXiv items).

Usage: python -I scripts/make_bib.py paper/references.bib
- arXiv entries: metadata pulled from the arXiv API at build time.
- A published venue is attached only when the authors' own arXiv comment/journal-ref states it,
  or when Crossref resolves the DOI (checked 2026-11-02).
- DOI-only entries: metadata copied from Crossref (checked 2026-11-02).
- Web entries: official URLs; sites that block scripted access carry `verifiedmanually`.
"""
from __future__ import annotations

import re
import sys
import urllib.request

ARXIV = {
    "angelopoulos2022crc": "2208.02814", "gibbs2021aci": "2106.00170", "chen2023frugalgpt": "2305.05176",
    "dou2026conformalcascade": "2607.25018", "guo2026routenlp": "2604.23577", "opoku2026promptshift": "2606.15964",
    "saeed2026shiftjev": "2610.34329", "shihab2026continual": "2604.23987", "rida2026tracer": "2604.14531",
    "dolphin2026eightk": "2607.08346", "wang2026cfpbrelief": "2606.22664", "liang2025llmwriting": "2502.09747",
    "li2026triageaudit": "2605.20956", "tibshirani2019weighted": "1904.06019", "warner2024modernbert": "2412.13663",
    "wang2020minilm": "2002.10957", "yang2025qwen3": "2505.09388", "lipton2018bbse": "1802.03916",
    "rabanser2019failing": "1810.11953", "yao2022wildtime": "2211.14238", "madras2018predict": "1711.06664",
    "mozannar2020defer": "2006.01862", "northcutt2021labelerrors": "2103.14749", "akhtar2024croissant": "2403.19546",
    "koh2021wilds": "2012.07421", "santurkar2020breeds": "2008.04859", "zaffran2022agaci": "2202.07282",
    "ding2023classcond": "2306.09335", "liang2024selectiveshift": "2405.05160", "zollo2024prc": "2311.13628",
    "lazaridou2021mindgap": "2102.01951",
}
VENUE = {  # stated by the authors on arXiv (comment / journal-ref)
    "angelopoulos2022crc": ("International Conference on Learning Representations (ICLR)", "2024"),
    "chen2023frugalgpt": ("Transactions on Machine Learning Research", "2024"),
    "guo2026routenlp": ("Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics: Industry Track", "2026"),
    "lipton2018bbse": ("International Conference on Machine Learning (ICML)", "2018"),
    "rabanser2019failing": ("Advances in Neural Information Processing Systems 32", "2019"),
    "yao2022wildtime": ("Advances in Neural Information Processing Systems 35, Datasets and Benchmarks Track", "2022"),
    "madras2018predict": ("Advances in Neural Information Processing Systems 31", "2018"),
    "mozannar2020defer": ("International Conference on Machine Learning (ICML)", "2020"),
    "northcutt2021labelerrors": ("Advances in Neural Information Processing Systems 34, Datasets and Benchmarks Track", "2021"),
    "akhtar2024croissant": ("Advances in Neural Information Processing Systems 37, Datasets and Benchmarks Track", "2024"),
    "liang2024selectiveshift": ("Transactions on Machine Learning Research", "2024"),
    "zollo2024prc": ("International Conference on Learning Representations (ICLR)", "2024"),
    "lazaridou2021mindgap": ("Advances in Neural Information Processing Systems 34", "2021"),
}

MANUAL = r"""

@article{bates2021rcps,
  author = {Bates, Stephen and Angelopoulos, Anastasios and Lei, Lihua and Malik, Jitendra and Jordan, Michael},
  title = {Distribution-free, Risk-controlling Prediction Sets},
  journal = {Journal of the ACM},
  volume = {68}, number = {6}, pages = {1--34}, year = {2021},
  doi = {10.1145/3478535}
}

@article{angelopoulos2021ltt,
  author = {Angelopoulos, Anastasios N. and Bates, Stephen and Cand{\`e}s, Emmanuel J. and Jordan, Michael I. and Lei, Lihua},
  title = {Learn then test: Calibrating predictive algorithms to achieve risk control},
  journal = {The Annals of Applied Statistics},
  volume = {19}, number = {2}, pages = {1641--1662}, year = {2025},
  doi = {10.1214/24-AOAS1998}
}

@article{barber2023beyond,
  author = {Barber, Rina Foygel and Cand{\`e}s, Emmanuel J. and Ramdas, Aaditya and Tibshirani, Ryan J.},
  title = {Conformal prediction beyond exchangeability},
  journal = {The Annals of Statistics},
  volume = {51}, number = {2}, pages = {816--845}, year = {2023},
  doi = {10.1214/23-AOS2276}
}

@article{clopper1934binomial,
  author = {Clopper, C. J. and Pearson, E. S.},
  title = {The Use of Confidence or Fiducial Limits Illustrated in the Case of the Binomial},
  journal = {Biometrika},
  volume = {26}, number = {4}, pages = {404--413}, year = {1934},
  doi = {10.1093/biomet/26.4.404}
}

@inproceedings{geifman2017selective,
  author = {Geifman, Yonatan and El-Yaniv, Ran},
  title = {Selective Classification for Deep Neural Networks},
  booktitle = {Advances in Neural Information Processing Systems 30},
  pages = {4878--4887}, year = {2017},
  url = {https://proceedings.neurips.cc/paper/2017/hash/4a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html}
}

@inproceedings{reimers2019sbert,
  author = {Reimers, Nils and Gurevych, Iryna},
  title = {Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks},
  booktitle = {Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)},
  pages = {3980--3990}, year = {2019},
  doi = {10.18653/v1/D19-1410}
}

@inproceedings{lee2022dedup,
  author = {Lee, Katherine and Ippolito, Daphne and Nystrom, Andrew and Zhang, Chiyuan and Eck, Douglas and Callison-Burch, Chris and Carlini, Nicholas},
  title = {Deduplicating Training Data Makes Language Models Better},
  booktitle = {Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages = {8424--8445}, year = {2022},
  doi = {10.18653/v1/2022.acl-long.577}
}

@article{gebru2021datasheets,
  author = {Gebru, Timnit and Morgenstern, Jamie and Vecchione, Briana and Vaughan, Jennifer Wortman and Wallach, Hanna and Daum{\'e} III, Hal and Crawford, Kate},
  title = {Datasheets for datasets},
  journal = {Communications of the ACM},
  volume = {64}, number = {12}, pages = {86--92}, year = {2021},
  doi = {10.1145/3458723}
}

@article{kunsch1989bootstrap,
  author = {K{\"u}nsch, Hans R.},
  title = {The Jackknife and the Bootstrap for General Stationary Observations},
  journal = {The Annals of Statistics},
  volume = {17}, number = {3}, pages = {1217--1241}, year = {1989},
  doi = {10.1214/aos/1176347265}
}

@inproceedings{broder1997resemblance,
  author = {Broder, Andrei Z.},
  title = {On the resemblance and containment of documents},
  booktitle = {Proceedings. Compression and Complexity of SEQUENCES 1997},
  pages = {21--29}, year = {1997},
  doi = {10.1109/SEQUEN.1997.666900}
}

@article{gama2014drift,
  author = {Gama, Jo{\~a}o and {\v{Z}}liobait{\.e}, Indr{\.e} and Bifet, Albert and Pechenizkiy, Mykola and Bouchachia, Abdelhamid},
  title = {A survey on concept drift adaptation},
  journal = {ACM Computing Surveys},
  volume = {46}, number = {4}, pages = {1--37}, year = {2014},
  doi = {10.1145/2523813}
}

@article{holm1979,
  author = {Holm, Sture},
  title = {A Simple Sequentially Rejective Multiple Test Procedure},
  journal = {Scandinavian Journal of Statistics},
  volume = {6}, number = {2}, pages = {65--70}, year = {1979},
  url = {https://www.jstor.org/stable/4615733},
  verifiedmanually = {2026-11-02, JSTOR stable URL; no Crossref DOI exists}
}

@misc{cfpb2026ccdb,
  author = {{Consumer Financial Protection Bureau}},
  title = {Consumer Complaint Database},
  year = {2026},
  url = {https://www.consumerfinance.gov/data-research/consumer-complaints/},
  verifiedmanually = {2026-11-02, HTTP 200 via curl; the CDN returns 403 to Python urllib}
}

@misc{cfpb2026archive,
  author = {{Consumer Financial Protection Bureau}},
  title = {{CFPB} Consumer Complaint Database Narratives Archive ({FOIA} Electronic Reading Room)},
  year = {2026},
  note = {21 export files, December 2011 to August 2026; accessed 2026-10-31},
  url = {https://www.consumerfinance.gov/foia-requests/foia-electronic-reading-room/cfpb-consumer-complaint-database-narratives-archive/},
  verifiedmanually = {2026-11-02, HTTP 200 via curl; the CDN returns 403 to Python urllib}
}

@misc{bls2025oews,
  author = {{U.S. Bureau of Labor Statistics}},
  title = {Occupational Employment and Wage Statistics, May 2025: Customer Service Representatives (43-4051)},
  year = {2026},
  url = {https://data.bls.gov/oesprofile/?year=2025&major_group=430000&occupation=434051&measure=01&areas=INDUSTRY,STATE,MSA},
  verifiedmanually = {2026-11-02, read in a browser: national median hourly wage 21.53 USD; the site blocks scripted access}
}

@article{johnson2023mimiciv,
  author = {Johnson, Alistair E. W. and Bulgarelli, Lucas and Shen, Lu and Gayles, Alvin and Shammout, Ayad and Horng, Steven and Pollard, Tom J. and Hao, Sicheng and Moody, Benjamin and Gow, Brian and Lehman, Li-wei H. and Celi, Leo A. and Mark, Roger G.},
  title = {MIMIC-IV, a freely accessible electronic health record dataset},
  journal = {Scientific Data},
  volume = {10}, number = {1}, pages = {1}, year = {2023},
  doi = {10.1038/s41597-022-01899-x}
}

@misc{hannun2023mlx,
  author = {Hannun, Awni and Digani, Jagrit and Katharopoulos, Angelos and Collobert, Ronan},
  title = {{MLX}: Efficient and Flexible Machine Learning on {Apple} Silicon},
  year = {2023},
  url = {https://github.com/ml-explore/mlx}
}
"""


def tex_escape(s: str) -> str:
    s = s.replace("&amp;", "&")
    return s.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("#", r"\#")


def main() -> None:
    out = sys.argv[1]
    ids = ",".join(ARXIV.values())
    xml = urllib.request.urlopen(f"https://export.arxiv.org/api/query?id_list={ids}&max_results={len(ARXIV) + 5}", timeout=90).read().decode()
    by_id = {}
    for e in xml.split("<entry>")[1:]:
        aid = re.search(r"abs/(.*?)(v\d+)?</id>", e).group(1)
        title = " ".join(re.search(r"<title>(.*?)</title>", e, re.S).group(1).split())
        authors = re.findall(r"<name>(.*?)</name>", e)
        year = re.search(r"<published>(\d{4})", e).group(1)
        by_id[aid] = (title, authors, year)
    parts = []
    for key, aid in ARXIV.items():
        title, authors, year = by_id[aid]
        auth = " and ".join(authors if len(authors) <= 12 else authors[:12] + ["others"])
        if key in VENUE:
            venue, vyear = VENUE[key]
            typ, vf = ("article", f"  journal = {{{venue}}},\n") if "Transactions" in venue else ("inproceedings", f"  booktitle = {{{venue}}},\n")
            parts.append(f"@{typ}{{{key},\n  author = {{{tex_escape(auth)}}},\n  title = {{{{{tex_escape(title)}}}}},\n{vf}"
                         f"  year = {{{vyear}}},\n  eprint = {{{aid}}},\n  archiveprefix = {{arXiv}},\n  url = {{https://arxiv.org/abs/{aid}}}\n}}\n")
        else:
            parts.append(f"@misc{{{key},\n  author = {{{tex_escape(auth)}}},\n  title = {{{{{tex_escape(title)}}}}},\n"
                         f"  year = {{{year}}},\n  eprint = {{{aid}}},\n  archiveprefix = {{arXiv}},\n  howpublished = {{arXiv:{aid}}},\n"
                         f"  url = {{https://arxiv.org/abs/{aid}}}\n}}\n")
    open(out, "w").write("\n".join(parts) + MANUAL)
    print(f"wrote {len(parts)} arXiv + manual entries to {out}")


if __name__ == "__main__":
    main()
