import sys


def main():
    claim = " ".join(sys.argv[1:]) or "No claim provided."
    print(f"[ClaimLens] Claim received: {claim}")
    print("[ClaimLens] Pipeline not yet implemented.")


if __name__ == "__main__":
    main()
