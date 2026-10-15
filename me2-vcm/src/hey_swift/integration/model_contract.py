"""Decode verified model indices without inventing their order or thresholds."""
from hey_swift.integration.router import INTENTS, SLOTS, Prediction


class ModelContract:
    def __init__(self, intent_labels, slot_labels):
        self.intent_labels = tuple(intent_labels)
        self.slot_labels = {intent: tuple(values) for intent, values in slot_labels.items()}
        if (len(set(self.intent_labels)) != len(self.intent_labels)
                or set(self.intent_labels) - (set(INTENTS) | {"OUT_OF_SCOPE"})
                or set(INTENTS) - set(self.intent_labels)):
            raise ValueError("Model intent labels must match all supported intents, with optional OUT_OF_SCOPE.")
        if set(self.slot_labels) != set(SLOTS):
            raise ValueError("Provide the selected export's label order for each slotted intent.")
        for intent, labels in self.slot_labels.items():
            if len(set(labels)) != len(labels) or set(labels) != set(SLOTS[intent]):
                raise ValueError(f"Model slots do not match supported values for {intent}.")

    def decode(self, intent_index, *, accepted, slot_index=None):
        if type(intent_index) is not int or not 0 <= intent_index < len(self.intent_labels):
            raise ValueError("Invalid intent output index.")
        if type(accepted) is not bool:
            raise ValueError("Acceptance must be an explicit boolean from upstream rejection validation.")
        intent = self.intent_labels[intent_index]
        if not accepted or intent == "OUT_OF_SCOPE":
            return Prediction(intent, accepted=False)
        if intent in SLOTS:
            labels = self.slot_labels[intent]
            if type(slot_index) is not int or not 0 <= slot_index < len(labels):
                raise ValueError("A valid slot output index is required.")
            return Prediction(intent, labels[slot_index], accepted=True)
        if slot_index is not None:
            raise ValueError("Fixed intents must not supply a slot index.")
        return Prediction(intent, accepted=True)
