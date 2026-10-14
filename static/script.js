const form = document.getElementById("sensor-form");
const button = document.getElementById("predict-btn");
const result = document.getElementById("result");
const condition = document.getElementById("condition");
const probabilities = document.getElementById("probabilities");
const message = document.getElementById("message");

form.addEventListener("submit", async (event) => {
    event.preventDefault();

    message.textContent = "";
    result.hidden = true;
    button.disabled = true;
    button.textContent = "Predicting...";

    try {
        const formData = new FormData(form);
        const payload = {};

        for (const [key, value] of formData.entries()) {
            payload[key] = Number(value);
        }

        const response = await fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(payload)
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(
                data.detail
                    ? JSON.stringify(data.detail)
                    : "Prediction failed."
            );
        }

        const conditionLabels = {
        "0": "Good Condition",
        "1": "Critical Condition"
    };

    condition.textContent =
        conditionLabels[String(data.prediction)] ??
        "Unknown Condition";

condition.style.color =
    String(data.prediction) === "0" ? "#22c55e" : "#ef4444";
        probabilities.replaceChildren();

        if (data.probabilities) {
            const heading = document.createElement("strong");
            heading.textContent = "Class probabilities";
            probabilities.appendChild(heading);

            for (const [label, probability] of
                Object.entries(data.probabilities)) {
                const line = document.createElement("div");
                line.textContent =
                    `Class ${label}: ${(probability * 100).toFixed(2)}%`;
                probabilities.appendChild(line);
            }
        }

        result.hidden = false;

    } catch (error) {
        message.textContent = "Error: " + error.message;
    } finally {
        button.disabled = false;
        button.textContent = "Predict Condition";
    }
});