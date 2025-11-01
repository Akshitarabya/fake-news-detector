document.addEventListener("DOMContentLoaded", () => {
    const checkButton = document.getElementById("checkButton");
    const resultDiv = document.getElementById("result");
    const newsInput = document.getElementById("newsInput");

    const API_URL = "https://fake-news-detector-3-ual0.onrender.com/predict";  // <== updated with /predict

    checkButton.addEventListener("click", async () => {
        const newsText = newsInput.value.trim();
        if (!newsText) {
            resultDiv.textContent = "⚠ Please enter some text!";
            resultDiv.style.color = "orange";
            return;
        }

        resultDiv.textContent = "🔍 Checking…";
        resultDiv.style.color = "gray";

        try {
            const response = await fetch(API_URL, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({ text: newsText })
            });

            if (!response.ok) {
                throw new Error(`Server responded with status ${response.status}`);
            }

            const data = await response.json();

            if (!data.prediction || data.confidence == null) {
                throw new Error("Unexpected response shape");
            }

            const conf = Number(data.confidence);
            const confDisplay = isNaN(conf) ? data.confidence : `${conf}%`;

            if (data.prediction === "FAKE") {
                resultDiv.textContent = `❌ Fake News (Confidence: ${confDisplay})`;
                resultDiv.style.color = "red";
            } else {
                resultDiv.textContent = `✅ Real News (Confidence: ${confDisplay})`;
                resultDiv.style.color = "limegreen";
            }

        } catch (error) {
            console.error("Error:", error);
            resultDiv.textContent = "⚠ Error connecting to the server!";
            resultDiv.style.color = "orange";
        }
    });
});
