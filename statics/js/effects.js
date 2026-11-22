
document.addEventListener("DOMContentLoaded", function () {
    // Create the floating leaves container.
    let container = document.getElementById("floating-leaves");

    if (!container) {
        container = document.createElement("div");
        container.id = "floating-leaves";
        document.body.appendChild(container);
    }

    Object.assign(container.style, {
        position: "fixed",
        inset: "0",
        width: "100%",
        height: "100%",
        overflow: "hidden",
        pointerEvents: "none",
        zIndex: "0"
    });

    // Leaf symbols and animation settings.
    const leafSymbols = ["🍃", "🍀", "🌿", "🍃"];
    const leaves = [];
    const leafCount = 18;

    for (let i = 0; i < leafCount; i++) {
        const leaf = document.createElement("span");

        leaf.textContent =
            leafSymbols[Math.floor(Math.random() * leafSymbols.length)];

        Object.assign(leaf.style, {
            position: "absolute",
            left: "0",
            top: "0",
            fontSize: (16 + Math.random() * 18) + "px",
            opacity: String(0.35 + Math.random() * 0.4),
            pointerEvents: "none",
            userSelect: "none",
            willChange: "transform"
        });

        container.appendChild(leaf);

        leaves.push({
            element: leaf,
            x: Math.random() * window.innerWidth,
            y: Math.random() * window.innerHeight,
            vx: (Math.random() - 0.5) * 0.6,
            vy: (Math.random() - 0.5) * 0.6,
            rotation: Math.random() * 360,
            rotationSpeed: (Math.random() - 0.5) * 1.5,
            drift: 0.2 + Math.random() * 0.4
        });
    }

    // Cursor position.
    let mouseX = -1000;
    let mouseY = -1000;
    const repelRadius = 150;
    const repelStrength = 3.5;

    document.addEventListener("mousemove", function (event) {
        mouseX = event.clientX;
        mouseY = event.clientY;
    });

    // Support touch screens.
    document.addEventListener("touchmove", function (event) {
        if (event.touches.length > 0) {
            mouseX = event.touches[0].clientX;
            mouseY = event.touches[0].clientY;
        }
    }, { passive: true });

    // Move leaves naturally and push them away from the cursor.
    function animateLeaves() {
        const width = window.innerWidth;
        const height = window.innerHeight;

        leaves.forEach(function (leaf) {
            const dx = leaf.x - mouseX;
            const dy = leaf.y - mouseY;
            const distance = Math.sqrt(dx * dx + dy * dy);

            if (distance < repelRadius && distance > 0) {
                const force =
                    (1 - distance / repelRadius) * repelStrength;

                leaf.vx += (dx / distance) * force;
                leaf.vy += (dy / distance) * force;
            }

            // Gentle natural floating motion.
            leaf.vx += (Math.random() - 0.5) * 0.025;
            leaf.vy += (Math.random() - 0.5) * 0.025;

            // Limit speed so the leaves remain smooth.
            const maxSpeed = 3;
            const speed = Math.sqrt(
                leaf.vx * leaf.vx + leaf.vy * leaf.vy
            );

            if (speed > maxSpeed) {
                leaf.vx = (leaf.vx / speed) * maxSpeed;
                leaf.vy = (leaf.vy / speed) * maxSpeed;
            }

            leaf.x += leaf.vx;
            leaf.y += leaf.vy;

            // Gradually slow the leaves after being pushed.
            leaf.vx *= 0.985;
            leaf.vy *= 0.985;

            // Add a small floating drift.
            leaf.x += Math.sin(leaf.y * 0.01) * leaf.drift;
            leaf.y += Math.cos(leaf.x * 0.01) * leaf.drift;

            leaf.rotation += leaf.rotationSpeed;

            // Wrap leaves around the screen edges.
            if (leaf.x < -40) leaf.x = width + 40;
            if (leaf.x > width + 40) leaf.x = -40;
            if (leaf.y < -40) leaf.y = height + 40;
            if (leaf.y > height + 40) leaf.y = -40;

            leaf.element.style.transform =
                `translate3d(${leaf.x}px, ${leaf.y}px, 0) ` +
                `translate(-50%, -50%) rotate(${leaf.rotation}deg)`;
        });

        requestAnimationFrame(animateLeaves);
    }

    animateLeaves();

    // Keep leaves inside the visible screen after resizing.
    window.addEventListener("resize", function () {
        leaves.forEach(function (leaf) {
            leaf.x = Math.min(leaf.x, window.innerWidth);
            leaf.y = Math.min(leaf.y, window.innerHeight);
        });
    });
});
