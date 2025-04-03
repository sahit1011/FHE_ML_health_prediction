// Main JavaScript file for the Secure Health Prediction app

document.addEventListener('DOMContentLoaded', function() {
    // Form validation
    const form = document.querySelector('form');
    if (form) {
        form.addEventListener('submit', function(event) {
            const age = document.getElementById('age');
            const systolicBP = document.getElementById('systolic_bp');
            const diastolicBP = document.getElementById('diastolic_bp');
            const bloodSugar = document.getElementById('blood_sugar');

            let isValid = true;

            // Age validation
            if (age.value < 25 || age.value > 75) {
                isValid = false;
                age.classList.add('is-invalid');
            } else {
                age.classList.remove('is-invalid');
            }

            // Systolic BP validation
            if (systolicBP.value < 90 || systolicBP.value > 180) {
                isValid = false;
                systolicBP.classList.add('is-invalid');
            } else {
                systolicBP.classList.remove('is-invalid');
            }

            // Diastolic BP validation
            if (diastolicBP.value < 60 || diastolicBP.value > 110) {
                isValid = false;
                diastolicBP.classList.add('is-invalid');
            } else {
                diastolicBP.classList.remove('is-invalid');
            }

            // Blood Sugar validation
            if (bloodSugar.value < 70 || bloodSugar.value > 150) {
                isValid = false;
                bloodSugar.classList.add('is-invalid');
            } else {
                bloodSugar.classList.remove('is-invalid');
            }

            if (!isValid) {
                event.preventDefault();
                alert('Please check the input values. They must be within the specified ranges.');
            }
        });
    }

    // Add loading indicator when form is submitted
    const submitBtn = document.querySelector('#submit-btn');
    const loadingIndicator = document.querySelector('#loading-indicator');

    if (submitBtn && loadingIndicator && form) {
        form.addEventListener('submit', function(event) {
            if (form.checkValidity()) {
                // Show loading indicator
                loadingIndicator.classList.remove('d-none');

                // Disable submit button
                submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Processing...';
                submitBtn.disabled = true;

                // Set a timeout to show a message if it's taking too long
                setTimeout(function() {
                    if (loadingIndicator.classList.contains('d-none') === false) {
                        const timeoutMsg = document.createElement('div');
                        timeoutMsg.className = 'alert alert-warning mt-3';
                        timeoutMsg.innerHTML = '<i class="fas fa-exclamation-triangle me-2"></i>This is taking longer than expected. FHE computations can be resource-intensive. Please continue to wait or try again later.';
                        loadingIndicator.appendChild(timeoutMsg);
                    }
                }, 15000); // Show message after 15 seconds
            }
        });
    }
});
