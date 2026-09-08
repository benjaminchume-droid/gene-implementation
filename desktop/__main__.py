from .runtime import DesktopRuntime


def main():
    runtime = DesktopRuntime()

    print("GENE DESKTOP RUNTIME")
    print(runtime.status())


if __name__ == "__main__":
    main()
