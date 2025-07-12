// Utility function to clear messages

function generateSimpleUUID() {
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, function (c) {
    const r = (Math.random() * 16) | 0;
    const v = c === "x" ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}

sessionStorage.setItem("sessionId", generateSimpleUUID());

function clearMessages() {
  document.getElementById("errorMsg").textContent = "";
  document.getElementById("successMsg").textContent = "";
}

// Load status and history on page load
window.onload = function () {
  console.log(
    "Page loaded. Checking MongoDB summaries status and loading Q&A history."
  );
  checkCombinedSummaryStatus();
  const userId = localStorage.getItem("user_id");
  if (userId) loadQAHistory();
};

async function summarizeFile() {
  console.log("Summarize File button clicked.");
  const fileInput = document.getElementById("fileInput");
  const file = fileInput.files[0];
  const summaryType = document.getElementById("summaryType").value;

  clearMessages();
  document.getElementById("loadingSection").style.display = "block";
  document.getElementById("resultSection").style.display = "none";
  document.getElementById("answerDisplay").textContent = "";
  document.getElementById("questionInput").value = "";

  if (!file) {
    document.getElementById("errorMsg").textContent = "Please upload a file.";
    document.getElementById("loadingSection").style.display = "none";
    console.log("No file selected for summarization.");
    return;
  }

  const formData = new FormData();
  formData.append("file", file);
  formData.append("summary_type", summaryType);
  console.log(
    `Attempting to summarize file: ${file.name} (Type: ${summaryType})`
  );

  try {
    const userId = localStorage.getItem("user_id");
    const response = await axios.post(
      "http://127.0.0.1:5005/process",
      formData,
      {
        headers: {
          userId: userId,
          "Content-Type": "multipart/form-data",
        },
      }
    );

    const { summary_text, security_mode, summary_id } = response.data;

    document.getElementById("summaryText").innerHTML = summary_text.replace(
      /\n/g,
      "<br>"
    );
    document.getElementById("resultSection").style.display = "block";
    document.getElementById(
      "successMsg"
    ).textContent = `Summary generated securely and stored in MongoDB! (ID: ${summary_id})`;

    checkCombinedSummaryStatus(); // Update status after new summary is added
  } catch (error) {
    console.error("Error during summarization:", error);
    document.getElementById("errorMsg").textContent =
      "Error processing file: " +
      (error?.response?.data?.error || error.message);
  } finally {
    document.getElementById("loadingSection").style.display = "none";
    fileInput.value = ""; // Clear file input
  }
}

async function askQuestion() {
  console.log("Ask Question button clicked.");
  const question = document.getElementById("questionInput").value.trim();
  const answerDisplay = document.getElementById("answerDisplay");
  const askQuestionBtn = document.getElementById("askQuestionBtn");
  const questionInput = document.getElementById("questionInput");
  const qaLoading = document.getElementById("qaLoading");

  clearMessages();

  if (!question) {
    document.getElementById("errorMsg").textContent =
      "Please enter a question.";
    console.log("No question entered.");
    return;
  }

  // Show loading
  qaLoading.style.display = "block";
  answerDisplay.textContent = "";
  console.log(`Sending question: "${question}" to backend.`);

  try {
    const userId = localStorage.getItem("user_id");
    const sessionId = sessionStorage.getItem("sessionId");
    const response = await axios.post(
      `http://127.0.0.1:5005/generate-answer`,
      {
        input: question,
        sessionId,
      },
      {
        headers: {
          userId: userId,
        },
      }
    );
    console.log("Q&A response:", response.data);

    if (response.data.answer) {
      answerDisplay.textContent = response.data.answer;
      document.getElementById(
        "successMsg"
      ).textContent = `Question answered successfully! Session ID: ${response.data.session_id}`;

      // Reload Q&A history to show the new session
      const userId = localStorage.getItem("user_id");
      if (userId) loadQAHistory();

      // Clear the question input
      questionInput.value = "";
    } else if (response.data.error) {
      answerDisplay.textContent = "Error: " + response.data.error;
      document.getElementById("errorMsg").textContent = response.data.error;
    } else {
      answerDisplay.textContent = "Sorry, no valid answer was returned.";
      document.getElementById("errorMsg").textContent =
        "No answer received from the AI.";
    }
  } catch (error) {
    answerDisplay.textContent = "";
    console.error("Error getting answer:", error);
    document.getElementById("errorMsg").textContent =
      "Error getting answer: " +
      (error?.response?.data?.error || error.message);

    if (
      error.code === "ERR_NETWORK" ||
      error.message.includes("Network Error")
    ) {
      document.getElementById("errorMsg").textContent =
        "Network error: Could not connect to the backend. Please ensure the Flask server is running.";
    }
  } finally {
    qaLoading.style.display = "none";
  }
}

async function checkCombinedSummaryStatus() {
  const statusDiv = document.getElementById("combinedSummaryStatus");
  const questionInput = document.getElementById("questionInput");
  const answerDisplay = document.getElementById("answerDisplay");

  statusDiv.innerHTML =
    '<span class="mongodb-icon"></span> Checking MongoDB summaries status...';
  questionInput.placeholder = "Loading Q&A context from MongoDB...";
  answerDisplay.textContent = "";
  clearMessages();

  try {
    const userId = localStorage.getItem("user_id");
    const response = await axios.get(
      "http://127.0.0.1:5005/check-summaries-status",
      {
        headers: {
          userId: userId,
        },
      }
    );
    console.log("MongoDB summaries status:", response.data);

    if (response.data.exists && response.data.count > 0) {
      statusDiv.innerHTML = `<span class="mongodb-icon"></span> MongoDB contains <strong>${
        response.data.count
      }</strong> summaries (${(response.data.size / 1024).toFixed(
        1
      )} KB) - Ready for Q&A`;
      statusDiv.style.color = "#27ae60";
      questionInput.placeholder = "Type your question here...";
    } else {
      statusDiv.innerHTML =
        '<span class="mongodb-icon"></span> MongoDB is empty. Please process documents to enable Q&A.';
      statusDiv.style.color = "#e74c3c";
      questionInput.placeholder = "Process documents to enable Q&A...";
    }
  } catch (error) {
    console.error("Error checking MongoDB summaries status:", error);
    statusDiv.innerHTML =
      '<span class="mongodb-icon"></span> Error connecting to MongoDB. Make sure the Flask backend is running.';
    statusDiv.style.color = "#e74c3c";
    questionInput.placeholder = "Error connecting to backend for Q&A context.";
    document.getElementById("errorMsg").textContent =
      "Failed to connect to backend. Please ensure the server is running.";
  }
}

async function loadQAHistory() {
  const qaHistoryLoading = document.getElementById("qaHistoryLoading");
  const qaHistoryContent = document.getElementById("qaHistoryContent");

  qaHistoryLoading.style.display = "block";
  qaHistoryContent.innerHTML = "";

  try {
    const userId = localStorage.getItem("user_id");
    const response = await axios.get("http://127.0.0.1:5005/qa-sessions", {
      headers: {
        userId: userId,
      },
    });
    console.log("Q&A sessions loaded:", response.data);

    if (response.data.length === 0) {
      qaHistoryContent.innerHTML =
        '<p style="color: #666; text-align: center; padding: 20px;">No Q&A history yet. Ask questions to build your history!</p>';
    } else {
      displayQAHistory(response.data);
    }
  } catch (error) {
    console.error("Error loading Q&A history:", error);
    qaHistoryContent.innerHTML =
      '<p style="color: #e74c3c; text-align: center; padding: 20px;">Error loading Q&A history. Please check backend connection.</p>';
  } finally {
    qaHistoryLoading.style.display = "none";
  }
}

function displayQAHistory(sessions) {
  const qaHistoryContent = document.getElementById("qaHistoryContent");
  let historyHTML = "";

  sessions.forEach((session) => {
    const sessionDate = new Date(session.created_at).toLocaleDateString();
    const sessionTime = new Date(session.created_at).toLocaleTimeString();

    historyHTML += `
                <div class="qa-session" onclick="toggleQASession('${session._id}')">
                    <div class="qa-session-header">${session.session_name}</div>
                    <div class="qa-session-date">${sessionDate} at ${sessionTime} • ${session.qa_pairs.length} Q&A pairs</div>
                    <div class="qa-session-content" id="session-${session._id}">
            `;
    session.qa_pairs.forEach((pair, index) => {
      historyHTML += `
                    <div class="qa-pair">
                        <div class="qa-question">Q${index + 1}: ${
        pair.question
      }</div>
                        <div class="qa-answer">${pair.answer}</div>
                    </div>
                `;
    });

    historyHTML += `
                    </div>
                </div>
            `;
  });

  qaHistoryContent.innerHTML = historyHTML;
}

function toggleQASession(sessionId) {
  const sessionContent = document.getElementById(`session-${sessionId}`);
  if (sessionContent.style.display === "block") {
    sessionContent.style.display = "none";
  } else {
    sessionContent.style.display = "block";
  }
}

async function clearQAHistory() {
  console.log("Clear Q&A History button clicked.");
  if (
    !window.confirm(
      `Are you sure you want to clear ALL Q&A history from MongoDB? This action cannot be undone.`
    )
  ) {
    console.log("Clear Q&A history action cancelled by user.");
    return;
  }

  clearMessages();

  try {
    const userId = localStorage.getItem("user_id");
    const response = await axios.delete(
      "http://127.0.0.1:5005/clear-qa-history",
      { headers: { userId: userId } }
    );
    document.getElementById("successMsg").textContent = response.data.message;
    console.log(response.data.message);

    if (userId) loadQAHistory();
  } catch (error) {
    console.error("Error clearing Q&A history:", error);
    document.getElementById("errorMsg").textContent =
      "Error clearing Q&A history: " +
      (error?.response?.data?.error || error.message);
    if (
      error.code === "ERR_NETWORK" ||
      error.message.includes("Network Error")
    ) {
      document.getElementById("errorMsg").textContent =
        "Network error: Could not connect to the backend to clear Q&A history. Please ensure the Flask server is running.";
    }
  }
}

async function downloadCombinedSummaries() {
  console.log("Download combined summaries button clicked.");
  clearMessages();
  try {
    const userId = localStorage.getItem("user_id");
    const response = await axios.get(
      `http://127.0.0.1:5005/download-combined-summaries`,
      {
        responseType: "blob",
        headers: {
          userId: userId,
        },
      }
    );

    const url = window.URL.createObjectURL(new Blob([response.data]));
    const link = document.createElement("a");
    link.href = url;
    link.setAttribute("download", "summaries.json");
    document.body.appendChild(link);
    link.click();
    link.remove();
    window.URL.revokeObjectURL(url);

    document.getElementById(
      "successMsg"
    ).textContent = `Downloaded summaries from MongoDB successfully.`;
    console.log(`Successfully downloaded summaries from MongoDB.`);
  } catch (error) {
    console.error("Error downloading combined summaries file:", error);
    document.getElementById("errorMsg").textContent =
      "Error downloading summaries: " +
      (error?.response?.data?.error || error.message);
    if (
      error.code === "ERR_NETWORK" ||
      error.message.includes("Network Error")
    ) {
      document.getElementById("errorMsg").textContent =
        "Network error: Could not connect to the backend to download summaries. Please ensure the Flask server is running.";
    }
  }
}

async function clearAllSummaries() {
  console.log("Clear All Summaries button clicked.");
  if (
    !window.confirm(
      `Are you sure you want to clear ALL summaries from MongoDB? This action cannot be undone.`
    )
  ) {
    console.log("Clear action cancelled by user.");
    return;
  }

  clearMessages();

  try {
    const userId = localStorage.getItem("user_id");

    const response = await axios.delete(
      `http://127.0.0.1:5005/clear-all-summaries`,
      {
        headers: {
          userId: userId,
        },
      }
    );
    document.getElementById("successMsg").textContent = response.data.message;
    console.log(response.data.message);
    document.getElementById("summaryText").innerHTML = "";
    document.getElementById("answerDisplay").textContent = "";
    document.getElementById("questionInput").value = "";
    document.getElementById("resultSection").style.display = "none";

    checkCombinedSummaryStatus(); // Update status after clearing
  } catch (error) {
    console.error("Error clearing all summaries:", error);
    document.getElementById("errorMsg").textContent =
      "Error clearing summaries: " +
      (error?.response?.data?.error || error.message);
    if (
      error.code === "ERR_NETWORK" ||
      error.message.includes("Network Error")
    ) {
      document.getElementById("errorMsg").textContent =
        "Network error: Could not connect to the backend to clear summaries. Please ensure the Flask server is running.";
    }
  }
}

function openTab(tabId, event) {
  console.log(`Opening tab: ${tabId}`);
  document
    .querySelectorAll(".tab-content")
    .forEach((t) => t.classList.remove("active"));
  document
    .querySelectorAll(".tab-button")
    .forEach((b) => b.classList.remove("active"));

  document.getElementById(tabId).classList.add("active");
  if (event && event.currentTarget) {
    event.currentTarget.classList.add("active");
  }
}
