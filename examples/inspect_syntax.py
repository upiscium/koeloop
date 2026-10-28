"""Inspect the locally installed CPU dependency parser; never downloads a model."""

import argparse
import json

import spacy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", nargs="*", default=[
        "私が昨日、駅で会った人から聞いた話によると、明日は雨になるそうです。",
        "窓を開けると冷たい風が入ってきたので、私は上着を着てから外へ出かけました。",
        "私が昨日駅で会って詳しい話を聞いた、隣町に住んでいる友人が来ました。",
        "今日は天気がよかったので、家族と近くの公園まで歩いて出かけました。",
        "この製品の特徴は、軽さと丈夫さと価格の手頃さです。",
    ])
    args = parser.parse_args()
    spacy.require_cpu()
    nlp = spacy.load("ja_core_news_sm", exclude=["ner"])
    for text in args.text:
        doc = nlp(text)
        print(json.dumps({"text": text, "tokens": [
            {"i": t.i, "start": t.idx, "text": t.text, "pos": t.pos_, "tag": t.tag_,
             "dep": t.dep_, "head": t.head.i, "head_text": t.head.text}
            for t in doc
        ]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
