class GeneExecutor:

    def __init__(self, runtime):
        self.runtime = runtime

    def execute_step(self, step: dict, user_input: str):

        if step["type"] == "reasoning":
            return {
                "success": True,
                "type": "reasoning",
                "action": step.get("action"),
            }

        tool = step["tool"]

        kwargs = {}

        if tool == "filesystem.list":
            kwargs["path"] = "gene"

        elif tool == "filesystem.search":
            kwargs["root"] = "gene"
            kwargs["query"] = user_input

        elif tool == "browser.navigate":
            words = user_input.split()

            url = next(
                (
                    word for word in words
                    if word.startswith("http://")
                    or word.startswith("https://")
                ),
                None,
            )

            if not url:
                return {
                    "success": False,
                    "error": "No URL detected in request.",
                }

            kwargs["url"] = url

        elif tool == "apps.find":
            kwargs["name"] = user_input

        elif tool == "process.execute":
            return {
                "success": False,
                "requires_confirmation": True,
                "error": "Process execution requires an explicit execution boundary.",
            }

        elif tool == "filesystem.read":
            return {
                "success": False,
                "error": "A target file path is required.",
            }

        elif tool == "filesystem.write":
            return {
                "success": False,
                "error": "A target path and content are required.",
            }

        return self.runtime.execute(tool, **kwargs)
