const usersStatus = document.getElementById("usersStatus");

function setUsersStatus(message, isError = false) {
    if (!usersStatus) {
        return;
    }
    usersStatus.textContent = message;
    usersStatus.className = `alert ${isError ? "error" : "info"}`;
}

async function deleteUser(row, button) {
    const userId = row.dataset.userId;
    const userName = row.dataset.userName || "this user";

    if (!window.confirm(`Delete ${userName} and all of their saved samples?`)) {
        return;
    }

    const originalLabel = button.textContent;
    button.disabled = true;
    button.textContent = "Deleting...";
    setUsersStatus(`Deleting ${userName} and retraining the model...`, false);

    try {
        const response = await fetch(`/users/${userId}/delete`, { method: "POST" });
        const data = await response.json();

        if (!response.ok || data.success === false) {
            throw new Error(data.message || "Delete failed.");
        }

        row.remove();
        setUsersStatus(data.message || `${userName} deleted successfully.`, false);

        const tbody = document.querySelector("tbody");
        if (tbody && tbody.querySelectorAll("tr[data-user-id]").length === 0) {
            tbody.innerHTML = '<tr><td colspan="6">No users enrolled yet.</td></tr>';
        }
    } catch (error) {
        console.error(error);
        setUsersStatus(error.message || "Delete failed.", true);
        button.disabled = false;
        button.textContent = originalLabel;
        return;
    }
}

document.querySelectorAll(".delete-user-btn").forEach((button) => {
    button.addEventListener("click", () => {
        const row = button.closest("tr[data-user-id]");
        if (!row) {
            return;
        }
        deleteUser(row, button);
    });
});
