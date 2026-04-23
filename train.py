from db import init_db
from face_utils import train_pca_model


def main():
    init_db()
    result = train_pca_model()
    metrics = result.get("metrics", {})
    print("Training completed successfully.")
    print(f"Registered users: {len(result.get('user_profiles', {}))}")
    print(f"Samples used: {int(metrics.get('samples', 0))}")
    print(f"Training accuracy: {metrics.get('accuracy', 0)}%")


if __name__ == "__main__":
    main()
