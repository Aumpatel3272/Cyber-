import argparse
from .data import generate_synthetic
from .train import train_model
from .evaluate import evaluate


def main():
    parser = argparse.ArgumentParser(prog="threat-ml")
    sub = parser.add_subparsers(dest="cmd")

    g = sub.add_parser("generate")
    g.add_argument("--out", default="data/dataset.csv")
    g.add_argument("--n", type=int, default=5000)

    t = sub.add_parser("train")
    t.add_argument("--data", required=True)
    t.add_argument("--out", default="models/model.joblib")

    e = sub.add_parser("evaluate")
    e.add_argument("--data", required=True)
    e.add_argument("--model", required=True)

    args = parser.parse_args()
    if args.cmd == "generate":
        generate_synthetic(args.out, n_samples=args.n)
        print("Generated", args.out)
    elif args.cmd == "train":
        p = train_model(args.data, out_path=args.out)
        print("Saved model to", p)
    elif args.cmd == "evaluate":
        out = evaluate(args.data, args.model)
        print(out["report"])
        print(f"\nAverage Threat Confidence: {out['average_threat_confidence']:.4f}")
        if out["feature_importance"]:
            print("\nTop 10 Important Features:")
            sorted_features = sorted(out["feature_importance"].items(), key=lambda x: x[1], reverse=True)
            for feature, importance in sorted_features[:10]:
                print(f"  {feature}: {importance:.4f}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
