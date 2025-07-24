from tkinter import *
import math

# Activation functions
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x):
    sx = sigmoid(x)
    return sx * (1 - sx)

# Global dictionary to hold intermediate values
EXPLAIN = {}

class WelcomeScreen(Frame):
    def __init__(self, parent, controller):
        Frame.__init__(self, parent, bg="#ffe6f0")
        Label(self, text="✨ Let's Learn Backpropagation ✨", font=("Georgia", 24, "bold"), bg="#ffe6f0", fg="#cc3399").pack(pady=100)
        Button(self, text="Start", font=("Comic Sans MS", 14), bg="#ffb6c1", command=lambda: controller.show_frame(CustomInputScreen)).pack()

class CustomInputScreen(Frame):
    def __init__(self, parent, controller):
        Frame.__init__(self, parent, bg="#ffe6f0")
        Label(self, text="🧾 Enter Inputs & Weights", font=("Georgia", 22), bg="#ffe6f0", fg="#cc3399").pack(pady=20)

        self.entries = {}
        fields = ["x1", "x2", "w11", "w21", "w12", "w22", "w13", "w23", "target"]

        for field in fields:
            frame = Frame(self, bg="#ffe6f0")
            frame.pack(pady=5)
            Label(frame, text=f"{field}:", font=("Comic Sans MS", 12), bg="#ffe6f0").pack(side=LEFT)
            entry = Entry(frame)
            entry.pack(side=LEFT)
            self.entries[field] = entry

        def save_inputs():
            for field in fields:
                try:
                    EXPLAIN[field] = float(self.entries[field].get())
                except:
                    EXPLAIN[field] = 0.0
            controller.refresh_all()
            controller.show_frame(DiagramScreen)

        Button(self, text="Next →", font=("Comic Sans MS", 12), bg="#ffb6c1", command=save_inputs).pack(pady=20)

class DiagramScreen(Frame):
    def __init__(self, parent, controller):
        Frame.__init__(self, parent, bg="#fff0f5")
        self.controller = controller
        self.canvas = Canvas(self, width=600, height=400, bg="#fff0f5", highlightthickness=0)
        self.canvas.pack(pady=10)
        self.desc = Label(self, text="", font=("Comic Sans MS", 13), bg="#fff0f5", wraplength=600)
        self.desc.pack(padx=20, pady=10)
        Button(self, text="Next →", font=("Comic Sans MS", 14), bg="#ffb6c1",
               command=lambda: controller.show_frame(InputExplanationScreen)).pack(pady=20)

    def refresh(self):
        self.canvas.delete("all")
        if 'x1' not in EXPLAIN:
            self.desc.config(text="Error: Inputs not set. Please go back and enter them first.")
            return

        x1, x2 = EXPLAIN['x1'], EXPLAIN['x2']
        w11, w21 = EXPLAIN['w11'], EXPLAIN['w21']
        w12, w22 = EXPLAIN['w12'], EXPLAIN['w22']
        w13, w23 = EXPLAIN['w13'], EXPLAIN['w23']

        self.canvas.create_oval(50, 150, 100, 200, fill="lightblue")  # Input x1
        self.canvas.create_oval(50, 250, 100, 300, fill="lightblue")  # Input x2
        self.canvas.create_text(75, 140, text=f"x1 = {x1}")
        self.canvas.create_text(75, 310, text=f"x2 = {x2}")

        self.canvas.create_oval(250, 150, 300, 200, fill="lightgreen")  # Hidden1
        self.canvas.create_oval(250, 250, 300, 300, fill="lightgreen")  # Hidden2

        self.canvas.create_oval(450, 200, 500, 250, fill="salmon")  # Output

        self.canvas.create_line(100, 175, 250, 175)  # x1 to H1
        self.canvas.create_text(175, 165, text=f"w11={w11}")
        self.canvas.create_line(100, 175, 250, 275)  # x1 to H2
        self.canvas.create_text(175, 230, text=f"w12={w12}")

        self.canvas.create_line(100, 275, 250, 175)  # x2 to H1
        self.canvas.create_text(175, 200, text=f"w21={w21}")
        self.canvas.create_line(100, 275, 250, 275)  # x2 to H2
        self.canvas.create_text(175, 265, text=f"w22={w22}")

        self.canvas.create_line(300, 175, 450, 225)  # H1 to Output
        self.canvas.create_text(375, 185, text=f"w13={w13}")
        self.canvas.create_line(300, 275, 450, 225)  # H2 to Output
        self.canvas.create_text(375, 265, text=f"w23={w23}")

        self.desc.config(text="XOR logic: Output is 1 when inputs differ. This network structure helps us visualize backpropagation clearly!")

class InputExplanationScreen(Frame):
    def __init__(self, parent, controller):
        Frame.__init__(self, parent, bg="#ffe6f0")
        Label(self, text="🌟 Inputs to the Network", font=("Georgia", 20), bg="#ffe6f0", fg="#cc3399").pack(pady=10)
        self.explanation = Label(self, text="", font=("Comic Sans MS", 14), bg="#ffe6f0")
        self.explanation.pack(padx=20, pady=10)
        Button(self, text="Next → Hidden Layer Input", font=("Comic Sans MS", 12), bg="#ffb6c1", command=lambda: controller.show_frame(HiddenInputScreen)).pack(pady=20)

    def refresh(self):
        x1, x2 = EXPLAIN['x1'], EXPLAIN['x2']
        self.explanation.config(text=f"x1 = {x1}, x2 = {x2}\nThese are the input values.")

class HiddenInputScreen(Frame):
    def __init__(self, parent, controller):
        Frame.__init__(self, parent, bg="#fff0f5")
        Label(self, text="🔧 Hidden Layer Weighted Sum", font=("Georgia", 20), bg="#fff0f5", fg="#cc3399").pack(pady=10)
        self.label = Label(self, text="", font=("Comic Sans MS", 13), bg="#fff0f5")
        self.label.pack(padx=20, pady=10)
        Button(self, text="Next → Activation", font=("Comic Sans MS", 12), bg="#ffb6c1", command=lambda: controller.show_frame(ActivationScreen)).pack(pady=20)

    def refresh(self):
        x1, x2 = EXPLAIN['x1'], EXPLAIN['x2']
        w11, w21 = EXPLAIN['w11'], EXPLAIN['w21']
        w12, w22 = EXPLAIN['w12'], EXPLAIN['w22']
        z3 = x1 * w11 + x2 * w21
        z4 = x1 * w12 + x2 * w22
        EXPLAIN['z3'], EXPLAIN['z4'] = z3, z4
        self.label.config(text=f"Z3 = {z3:.4f}, Z4 = {z4:.4f}\nThese are sums for hidden neurons.")

class ActivationScreen(Frame):
    def __init__(self, parent, controller):
        Frame.__init__(self, parent, bg="#ffe6f0")
        Label(self, text="⚡ Activation Function (Sigmoid)", font=("Georgia", 20), bg="#ffe6f0", fg="#cc3399").pack(pady=10)
        self.label = Label(self, text="", font=("Comic Sans MS", 13), bg="#ffe6f0")
        self.label.pack(padx=20, pady=10)
        Button(self, text="Next → Output Calculation", font=("Comic Sans MS", 12), bg="#ffb6c1", command=lambda: controller.show_frame(OutputScreen)).pack(pady=20)

    def refresh(self):
        z3, z4 = EXPLAIN['z3'], EXPLAIN['z4']
        y3 = sigmoid(z3)
        y4 = sigmoid(z4)
        EXPLAIN['y3'], EXPLAIN['y4'] = y3, y4
        self.label.config(text=f"Y3 = sigmoid({z3:.4f}) = {y3:.4f}, Y4 = sigmoid({z4:.4f}) = {y4:.4f}")

class OutputScreen(Frame):
    def __init__(self, parent, controller):
        Frame.__init__(self, parent, bg="#fff0f5")
        Label(self, text="🔁 Output Layer Calculation", font=("Georgia", 20), bg="#fff0f5", fg="#cc3399").pack(pady=10)
        self.label = Label(self, text="", font=("Comic Sans MS", 13), bg="#fff0f5")
        self.label.pack(padx=20, pady=10)
        Button(self, text="Next → Gradient Calculation", font=("Comic Sans MS", 12), bg="#ffb6c1", command=lambda: controller.show_frame(LossScreen)).pack(pady=20)

    def refresh(self):
        y3, y4 = EXPLAIN['y3'], EXPLAIN['y4']
        w13, w23 = EXPLAIN['w13'], EXPLAIN['w23']
        z5 = y3 * w13 + y4 * w23
        y5 = sigmoid(z5)
        EXPLAIN['z5'], EXPLAIN['y5'] = z5, y5
        self.label.config(text=f"Z5 = {z5:.4f}, Y5 = sigmoid({z5:.4f}) = {y5:.4f}\nFinal prediction.")

class LossScreen(Frame):
    def __init__(self, parent, controller):
        Frame.__init__(self, parent, bg="#ffe6f0")
        Label(self, text="💔 Loss Calculation", font=("Georgia", 20), bg="#ffe6f0", fg="#cc3399").pack(pady=10)
        self.label = Label(self, text="", font=("Comic Sans MS", 13), bg="#ffe6f0")
        self.label.pack(padx=20, pady=10)
        Button(self, text="🎯 Done!", font=("Comic Sans MS", 12), bg="#ffb6c1", command=lambda: controller.show_frame(GradientScreen)).pack(pady=20)

    def refresh(self):
        y5 = EXPLAIN['y5']
        target = EXPLAIN['target']
        loss = target - y5
        EXPLAIN['loss'] = loss
        self.label.config(text=f"Loss = Target - Output = {target} - {y5:.4f} = {loss:.4f}")

class GradientScreen(Frame):
    def __init__(self, parent, controller):
        Frame.__init__(self, parent, bg="#ffe6f0")
        Label(self, text="📐 Gradient Calculation", font=("Georgia", 20), bg="#ffe6f0", fg="#cc3399").pack(pady=10)
        self.label = Label(self, text="", font=("Comic Sans MS", 13), bg="#ffe6f0")
        self.label.pack(padx=20, pady=10)
        Button(self, text="Next → Weight Update", font=("Comic Sans MS", 12), bg="#ffb6c1", command=lambda: controller.show_frame(WeightUpdateScreen)).pack(pady=20)

    def refresh(self):
        y5 = EXPLAIN['y5']
        target = EXPLAIN['target']
        error = target - y5
        dL_dy5 = 2 * error
        dy5_dz5 = sigmoid_derivative(EXPLAIN['z5'])
        dz5_dw13 = EXPLAIN['y3']
        dz5_dw23 = EXPLAIN['y4']
        EXPLAIN['dL_dw13'] = dL_dy5 * dy5_dz5 * dz5_dw13
        EXPLAIN['dL_dw23'] = dL_dy5 * dy5_dz5 * dz5_dw23
        self.label.config(text=f"Gradient w13 = {EXPLAIN['dL_dw13']:.4f}, w23 = {EXPLAIN['dL_dw23']:.4f}")

class WeightUpdateScreen(Frame):
    def __init__(self, parent, controller):
        Frame.__init__(self, parent, bg="#fff0f5")
        Label(self, text="🔧 Weight Update (Gradient Descent)", font=("Georgia", 20), bg="#fff0f5", fg="#cc3399").pack(pady=10)
        self.label = Label(self, text="", font=("Comic Sans MS", 13), bg="#fff0f5")
        self.label.pack(padx=20, pady=10)
        Button(self, text="Next → Loss Calculation", font=("Comic Sans MS", 12), bg="#ffb6c1", command=lambda: controller.show_frame(SummaryScreen)).pack(pady=20)

    def refresh(self):
        lr = 0.1
        for key in ['w13', 'w23']:
            EXPLAIN[key] -= lr * EXPLAIN[f'dL_d{key}']
        self.label.config(text=f"Updated w13 = {EXPLAIN['w13']:.4f}, w23 = {EXPLAIN['w23']:.4f}")

class SummaryScreen(Frame):
    def __init__(self, parent, controller):
        Frame.__init__(self, parent, bg="#fff0f5")
        Label(self, text="📚 What Did We Learn?", font=("Georgia", 24, "bold"), bg="#fff0f5", fg="#cc3399").pack(pady=10)
        summary = (
            "1. Inputs were fed into the network.\n"
            "2. Weighted sums for hidden neurons were calculated.\n"
            "3. Sigmoid activation applied.\n"
            "4. Final output prediction made.\n"
            "5. Gradients were computed.\n"
            "6. Weights updated using gradient descent.\n"
        )
        Label(self, text=summary, font=("Comic Sans MS", 14), bg="#fff0f5", justify=LEFT).pack(padx=20, pady=20)
        Button(self, text="🏁 Restart", font=("Comic Sans MS", 12), bg="#ffb6c1", command=lambda: controller.show_frame(WelcomeScreen)).pack(pady=10)

class BrainspellApp(Tk):
    def __init__(self):
        Tk.__init__(self)
        self.title("Brainspell: Backpropagation Learner")
        self.geometry("960x680")
        self.configure(bg="#ffe6f0")

        container = Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}
        self.screens = (
            WelcomeScreen,
            CustomInputScreen,
            DiagramScreen,
            InputExplanationScreen,
            HiddenInputScreen,
            ActivationScreen,
            OutputScreen,
            LossScreen,
            GradientScreen,
            WeightUpdateScreen,
            SummaryScreen
        )

        for F in self.screens:
            frame = F(container, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(WelcomeScreen)

    def show_frame(self, cont):
        frame = self.frames[cont]
        if hasattr(frame, 'refresh'):
            frame.refresh()
        frame.tkraise()

    def refresh_all(self):
        for frame in self.frames.values():
            if hasattr(frame, 'refresh'):
                frame.refresh()

if __name__ == "__main__":
    app = BrainspellApp()
    app.mainloop()
