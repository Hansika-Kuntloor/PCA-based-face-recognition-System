from db import init_db
from face_utils import train_model_from_database


def main() -> None:
    init_db()
    result = train_model_from_database()
    metrics = result.get("metrics", {})
    print("Training completed successfully.")
    print(f"Registered users: {result.get('trained_users', 0)}")
    print(f"Samples used: {int(metrics.get('samples', 0))}")
    print(f"Training accuracy: {metrics.get('accuracy', 0)}%")
    print(f"False acceptance rate: {metrics.get('false_accept_rate', 0)}%")
    print(f"False rejection rate: {metrics.get('false_reject_rate', 0)}%")


if __name__ == "__main__":
    main()
