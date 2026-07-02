console.log("Dashboard JS loaded successfully!");

function updateDashboard() {
    const chart = document.getElementById("chart");
    chart.style.opacity = 0.3;

    fetch("/dashboard_data")
        .then(res => res.json())
        .then(data => {
            document.getElementById("drowsy_count").innerText = data.total_drowsy;
            document.getElementById("yawn_count").innerText = data.total_yawn;
            chart.src = "data:image/png;base64," + data.chart;
            chart.style.opacity = 1;
        })
        .catch(err => console.error(err));
}

document.getElementById("last_updated").innerText =
    "Last updated: " + new Date().toLocaleTimeString();

    function flash(id) {
    const el = document.getElementById(id);
    el.style.transition = "background 0.4s";
    el.style.background = "#ffecb3";
    setTimeout(() => el.style.background = "transparent", 400);
}

document.getElementById("drowsy_count").innerText = data.total_drowsy;
flash("drowsy_count");
