"""Deterministic English commands mapped to validated, source-relative time edits."""

import re
from dataclasses import asdict, dataclass, replace


class CommandError(ValueError):
    """A command needs clarification before any motion is changed."""


@dataclass(frozen=True)
class EditPlan:
    speed: float = 1.0
    start: float | None = None
    end: float | None = None
    transition: float = 0.0

    def validate(self, duration):
        from retime import source_times

        if not 0.25 <= self.speed <= 4:
            raise CommandError("Choose a speed between 0.25x and 4x.")
        try:
            source_times(
                2, 1 / duration, 30, self.speed, self.start, self.end, self.transition
            )
        except ValueError as error:
            raise CommandError(str(error)) from error
        return self

    def to_dict(self):
        return asdict(self)

    def describe(self):
        scope = (
            "the whole motion"
            if self.start is None
            else f"source seconds {self.start:g}-{self.end:g}"
        )
        ramp = (
            f", with {self.transition:g}s smooth transitions" if self.transition else ""
        )
        return f"{self.speed:g}x speed over {scope}{ramp}"


@dataclass(frozen=True)
class Command:
    action: str
    plan: EditPlan | None = None


NUMBER = r"(?:\d+(?:\.\d+)?|\.\d+)"


def parse_command(message, current, duration):
    """Absolute x-speeds use the original motion; percentages modify the current speed."""
    text = message.strip().lower().replace("\u00d7", "x").replace("\u2013", "-")
    if not text or len(text) > 500:
        raise CommandError("Enter an English motion command, up to 500 characters.")
    if re.fullmatch(r"(?:please )?undo(?: (?:that|last edit))?[.!]?", text):
        return Command("undo")
    if re.fullmatch(
        r"(?:please )?(?:reset(?: (?:the )?(?:motion|speed))?|normal speed|original speed|restore original)(?: please)?[.!]?",
        text,
    ):
        return Command("edit", EditPlan())
    if re.search(
        r"\b(jump|run|backward|reverse|turn|rotate|left|right|translate|stop|delete|upload|camera)\b",
        text,
    ):
        raise CommandError(
            "I can edit timing, not poses or direction. Try: Set speed to 1.5x."
        )
    if re.search(r"\b(then|afterwards|instead of)\b", text):
        raise CommandError("Send one timing edit at a time.")
    if re.search(r"\b(?:do not|don't|never|avoid)\b", text):
        raise CommandError(
            "Tell me the speed you want to apply, rather than a speed to avoid."
        )
    plan = current
    changed = False
    intervals = re.findall(
        rf"\bfrom\s+({NUMBER})\s*(?:seconds|sec|s)?\s+to\s+({NUMBER})\s*(?:seconds|sec|s)?",
        text,
    )
    if len(intervals) > 1:
        raise CommandError("Use one source interval per edit.")
    if intervals:
        a, b = map(float, intervals[0])
        plan = replace(plan, start=a, end=b)
        changed = True
    elif re.search(r"\b(from|between|interval|segment|only)\b", text):
        raise CommandError("Specify the interval as: from 0.5 to 1.5 seconds.")
    if re.search(r"\b(whole|entire|full)\b", text):
        if intervals:
            raise CommandError("Choose either the whole motion or a source interval.")
        plan = replace(plan, start=None, end=None, transition=0)
        changed = True
    factors = re.findall(rf"(?<![\w.\-])({NUMBER})\s*x\b", text)
    percentages = re.findall(rf"(?<![\w.\-])({NUMBER})\s*%\s*(faster|slower)", text)
    words = re.findall(r"\b(twice as fast|double speed|half speed)\b", text)
    if len(factors) + len(percentages) + len(words) > 1:
        raise CommandError("Specify a single speed. For example: Set speed to 1.5x.")
    if factors:
        plan = replace(plan, speed=float(factors[0]))
        changed = True
    elif percentages:
        value, direction = percentages[0]
        factor = 1 + float(value) / 100 * (1 if direction == "faster" else -1)
        plan = replace(plan, speed=current.speed * factor)
        changed = True
    elif words:
        plan = replace(plan, speed=0.5 if words[0] == "half speed" else 2)
        changed = True
    transition = re.search(
        rf"\b(?:transition|ramp)(?:s)?(?:\s+(?:of|to))?\s+({NUMBER})\s*(?:seconds|sec|s)\b",
        text,
    )
    if transition:
        plan = replace(plan, transition=float(transition[1]))
        changed = True
    elif re.search(r"\b(?:no smoothing|remove smoothing|hard transition)\b", text):
        plan = replace(plan, transition=0)
        changed = True
    elif re.search(r"\b(smooth|smoother|smoothly)\b", text):
        a, b = (
            (0 if plan.start is None else plan.start),
            (duration if plan.end is None else plan.end),
        )
        plan = replace(plan, transition=min(0.25, (b - a) / 4))
        changed = True
    # Reject unparsed numeric values rather than silently reinterpreting their units.
    remainder = text
    for pattern in [
        rf"\bfrom\s+{NUMBER}\s*(?:seconds|sec|s)?\s+to\s+{NUMBER}\s*(?:seconds|sec|s)?",
        rf"(?<![\w.\-]){NUMBER}\s*x\b",
        rf"(?<![\w.\-]){NUMBER}\s*%\s*(?:faster|slower)",
        rf"\b(?:transition|ramp)(?:s)?(?:\s+(?:of|to))?\s+{NUMBER}\s*(?:seconds|sec|s)\b",
    ]:
        remainder = re.sub(pattern, "", remainder)
    if re.search(r"\d|[-+]\s*(?:\d|\.)", remainder) or not changed:
        raise CommandError(
            "Use an explicit speed, such as: Set speed to 1.5x. You can also say: Make it 20% slower."
        )
    allowed = set(
        "please can could would you i want like make it this that the a an to at speed walk walking move motion retime play playback faster slower fast slow as twice double half whole entire full clip for with and smooth smoother smoothly transition transitions ramp ramps of no remove smoothing hard seconds s sec use set change apply".split()
    )
    if set(re.findall(r"[a-z]+", remainder)) - allowed:
        raise CommandError(
            "I support timing commands such as: Set speed to 1.5x from 0.5 to 1.5 seconds. Send other edits separately."
        )
    return Command("edit", plan.validate(duration))
