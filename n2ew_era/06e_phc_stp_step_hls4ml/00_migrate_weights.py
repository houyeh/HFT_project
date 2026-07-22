from common_06e import migrate_weights, verify_equivalence


if __name__ == "__main__":
    migrate_weights()
    report = verify_equivalence()
    print(report.to_string(index=False))
