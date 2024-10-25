// Function: Analyze Search Intent
function analyzeIntent() {
    const query = document.getElementById('intent-query').value;
    fetch('/api/analyze-intent', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({query: query})
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('intent-output').innerHTML = `Answer: ${data.answer}`;
    });
}

// Function: Voice Search Optimization
function startVoiceRecognition() {
    var recognition = new webkitSpeechRecognition();
    recognition.onresult = function(event) {
        const transcript = event.results[0][0].transcript;
        document.getElementById('voice-output').innerHTML = `Recognized: ${transcript}`;
        fetch('/api/analyze-voice', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({query: transcript})
        })
        .then(response => response.json())
        .then(data => {
            document.getElementById('voice-output').innerHTML += `<br>Optimized: ${data.optimized}`;
        });
    }
    recognition.start();
}

// Function: Get Keyword Gaps
function getKeywordGaps() {
    const keywords = document.getElementById('keywords-input').value.split(',');
    fetch('/api/keyword-gaps', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({keywords: keywords})
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('gaps-output').innerHTML = `Suggested Keywords: ${data.suggested_keywords.join(', ')}`;
    });
}

// Function: Check Mobile Optimization
function checkMobileOptimization() {
    const url = document.getElementById('mobile-url').value;
    fetch('/api/mobile-optimization', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({url: url})
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('mobile-output').innerHTML = `URL: ${data.url}, Score: ${data.score}, Suggestions: ${data.suggestions}`;
    });
}
