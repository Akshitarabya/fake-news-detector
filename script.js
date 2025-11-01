document.addEventListener("DOMContentLoaded", () => {
    const checkButton = document.getElementById("checkButton");
    const resultDiv = document.getElementById("result");
    const newsInput = document.getElementById("newsInput");

    checkButton.addEventListener("click", async () => {
        const newsText = newsInput.value.trim();

        if (!newsText) {
            resultDiv.textContent = "⚠ Please enter some text!";
            resultDiv.style.color = "orange";
            return;
        }

        try {
            const response = await fetch("http://127.0.0.1:5000/predict", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({ text: newsText })
            });

            if (!response.ok) throw new Error("Server not responding");

            const data = await response.json();

            if (data.prediction === "FAKE") {
                resultDiv.textContent = `❌ Fake News (Confidence: ${data.confidence}%)`;
                resultDiv.style.color = "red";
            } else {
                resultDiv.textContent = `✅ Real News (Confidence: ${data.confidence}%)`;
                resultDiv.style.color = "limegreen";
            }
        } catch (error) {
            console.error("Error:", error);
            resultDiv.textContent = "⚠ Error connecting to the server!";
            resultDiv.style.color = "orange";
        }
    });
});