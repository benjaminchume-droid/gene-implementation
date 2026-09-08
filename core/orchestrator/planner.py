class GenePlanner:

    def create_plan(self, user_input: str, tools: list[dict]) -> list[dict]:
        text = user_input.lower()

        plan = []

        if any(word in text for word in [
            "read", "open", "inspect", "show", "look at"
        ]):
            plan.append({
                "type": "tool",
                "tool": "filesystem.read",
                "reason": "User appears to request file inspection.",
            })

        elif any(word in text for word in [
            "write", "create", "save", "edit", "update"
        ]):
            plan.append({
                "type": "tool",
                "tool": "filesystem.write",
                "reason": "User appears to request filesystem modification.",
            })

        elif any(word in text for word in [
            "list", "folder", "directory", "files"
        ]):
            plan.append({
                "type": "tool",
                "tool": "filesystem.list",
                "reason": "User appears to request directory inspection.",
            })

        elif any(word in text for word in [
            "search", "find"
        ]):
            plan.append({
                "type": "tool",
                "tool": "filesystem.search",
                "reason": "User appears to request search.",
            })

        elif any(word in text for word in [
            "browser", "website", "web page", "navigate"
        ]):
            plan.append({
                "type": "tool",
                "tool": "browser.navigate",
                "reason": "User appears to request browser navigation.",
            })

        elif any(word in text for word in [
            "app", "application", "installed"
        ]):
            plan.append({
                "type": "tool",
                "tool": "apps.find",
                "reason": "User appears to request application discovery.",
            })

        elif any(word in text for word in [
            "run", "execute", "command", "terminal"
        ]):
            plan.append({
                "type": "tool",
                "tool": "process.execute",
                "reason": "User appears to request process execution.",
            })

        else:
            plan.append({
                "type": "reasoning",
                "action": "respond",
                "reason": "No deterministic tool requirement detected.",
            })

        return plan
