from tkinter import *
from PIL import Image, ImageTk
import os, math

# Activation functions
def sigmoid(x):
    # Sigmoid activation: squashes input between 0 and 1
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x):
    # Derivative of sigmoid σ(x) is: σ(x) * (1 - σ(x))
    sx = sigmoid(x)
    return sx * (1 - sx)

# Store globals for step-by-step explanation
EXPLAIN = {}

class WelcomeScreen(Frame):
    def __init__(self, parent, controller):
        Frame.__init__(self, parent, bg="#ffe6f0")
        Label(self, text="✨ Let's Learn Backpropagation ✨", font=("Georgia", 24, "bold"), bg="#ffe6f0", fg="#cc3399").pack(pady=100)
        Button(self, text="Start Game", font=("Comic Sans MS", 14), bg="#ffb6c1", command=lambda: controller.show_frame(DiagramScreen)).pack()

class DiagramScreen(Frame):
    def __init__(self, parent, controller):
        Frame.__init__(self, parent, bg="#fff0f5")
        Label(self, text="📊 XOR Neural Network", font=("Georgia", 22, "bold"), bg="#fff0f5", fg="#cc3399").pack(pady=10)

        # Backpropagation diagram
        try:
            backprop_img_path = os.path.join("assets", "backprop.png")
            self.backprop_img = ImageTk.PhotoImage(Image.open(backprop_img_path).resize((400, 350)))
            Label(self, image=self.backprop_img, bg="#fff0f5").pack(pady=10)
        except:
            Label(self, text="(Couldn't load backprop diagram image)", bg="#fff0f5").pack()

        # XOR diagram
        try:
            xor_img_path = os.path.join("assets", "xor_diagramt.png")
            self.img = ImageTk.PhotoImage(Image.open(xor_img_path).resize((200, 150)))
            Label(self, image=self.img, bg="#fff0f5").pack(pady=10)
        except:
            Label(self, text="(Couldn't load XOR diagram image)", bg="#fff0f5").pack()

        # XOR logic explanation
        Label(
            self,
            text="The output is 1 only when the two inputs are different. Otherwise, it's 0.\n"
                 "It’s like saying, ‘either this or that, but not both!’",
            font=("Comic Sans MS", 14),
            bg="#fff0f5"
        ).pack(pady=10)

        Button(self, text="Next →", font=("Comic Sans MS", 14), bg="#ffb6c1",
               command=lambda: controller.show_frame(InputExplanationScreen)).pack(pady=20)
        
class InputExplanationScreen(Frame):
    def __init__(self, parent, controller):
        Frame.__init__(self, parent, bg="#ffe6f0")

        try:
            backprop_img_path = os.path.join("assets", "backprop.png")
            self.backprop_img = ImageTk.PhotoImage(Image.open(backprop_img_path).resize((400, 350)))
            Label(self, image=self.backprop_img, bg="#ffe6f0").pack(pady=10)
        except:
            Label(self, text="(Couldn't load backprop diagram image)", bg="#ffe6f0").pack()

        Label(self, text="🌟 Inputs to the Network", font=("Georgia", 20), bg="#ffe6f0", fg="#cc3399").pack(pady=10)

        inputs = [0.35, 0.7]
        EXPLAIN['x1'], EXPLAIN['x2'] = inputs

        explanation = f"x1 = {inputs[0]}\nx2 = {inputs[1]}\n\nThese values are inputs to the network.\nThey represent features from the dataset (like pixels in images, values in a table, etc)."

        Label(self, text=explanation, font=("Comic Sans MS", 14), bg="#ffe6f0", justify=LEFT).pack(padx=20, pady=10)
        Button(self, text="Next → Hidden Layer Input", font=("Comic Sans MS", 12), bg="#ffb6c1", command=lambda: controller.show_frame(HiddenInputScreen)).pack(pady=20)

class HiddenInputScreen(Frame):
    def __init__(self, parent, controller):
        Frame.__init__(self, parent, bg="#fff0f5")

        try:
            backprop_img_path = os.path.join("assets", "backprop.png")
            self.backprop_img = ImageTk.PhotoImage(Image.open(backprop_img_path).resize((400, 350)))
            Label(self, image=self.backprop_img, bg="#fff0f5").pack(pady=10)
        except:
            Label(self, text="(Couldn't load backprop diagram image)", bg="#fff0f5").pack()

        Label(self, text="🔧 Hidden Layer Weighted Sum", font=("Georgia", 20), bg="#fff0f5", fg="#cc3399").pack(pady=10)

        x1, x2 = EXPLAIN['x1'], EXPLAIN['x2'] = 0.35, 0.7
        w11, w21 = 0.2, 0.2
        w12, w22 = 0.3, 0.3
        z3 = x1 * w11 + x2 * w21
        z4 = x1 * w12 + x2 * w22

        EXPLAIN['z3'], EXPLAIN['z4'] = z3, z4
        explanation = f"Z3 = x1*w11 + x2*w21 = {x1}*{w11} + {x2}*{w21} = {z3:.4f}\nZ4 = x1*w12 + x2*w22 = {x1}*{w12} + {x2}*{w22} = {z4:.4f}\n\nThese are the raw input values received by the hidden neurons before activation."

        Label(self, text=explanation, font=("Comic Sans MS", 13), bg="#fff0f5", justify=LEFT).pack(padx=20, pady=10)
        Button(self, text="Next → Activation", font=("Comic Sans MS", 12), bg="#ffb6c1", command=lambda: controller.show_frame(ActivationScreen)).pack(pady=20)

class ActivationScreen(Frame):
    def __init__(self, parent, controller):
        Frame.__init__(self, parent, bg="#ffe6f0")

        try:
            backprop_img_path = os.path.join("assets", "backprop.png")
            self.backprop_img = ImageTk.PhotoImage(Image.open(backprop_img_path).resize((400, 350)))
            Label(self, image=self.backprop_img, bg="#ffe6f0").pack(pady=10)
        except:
            Label(self, text="(Couldn't load backprop diagram image)", bg="#ffe6f0").pack()

        Label(self, text="⚡ Activation Function (Sigmoid)", font=("Georgia", 20), bg="#ffe6f0", fg="#cc3399").pack(pady=10)

        z3, z4 = EXPLAIN['z3'], EXPLAIN['z4']
        y3 = sigmoid(z3)
        y4 = sigmoid(z4)
        EXPLAIN['y3'], EXPLAIN['y4'] = y3, y4

        explanation = f"Y3 = sigmoid({z3:.4f}) = {y3:.4f}\nY4 = sigmoid({z4:.4f}) = {y4:.4f}\n\nThe sigmoid function squashes the input between 0 and 1, mimicking real neuron behavior."

        Label(self, text=explanation, font=("Comic Sans MS", 13), bg="#ffe6f0", justify=LEFT).pack(padx=20, pady=10)
        Button(self, text="Next → Output Calculation", font=("Comic Sans MS", 12), bg="#ffb6c1", command=lambda: controller.show_frame(OutputScreen)).pack(pady=20)

class OutputScreen(Frame):
    def __init__(self, parent, controller):
        Frame.__init__(self, parent, bg="#fff0f5")

        try:
            backprop_img_path = os.path.join("assets", "backprop.png")
            self.backprop_img = ImageTk.PhotoImage(Image.open(backprop_img_path).resize((400, 350)))
            Label(self, image=self.backprop_img, bg="#fff0f5").pack(pady=10)
        except:
            Label(self, text="(Couldn't load backprop diagram image)", bg="#fff0f5").pack()

        Label(self, text="🔁 Final Output Calculation", font=("Georgia", 20), bg="#fff0f5", fg="#cc3399").pack(pady=10)

        y3, y4 = EXPLAIN['y3'], EXPLAIN['y4']
        w13, w23 = 0.3, 0.9
        z5 = y3 * w13 + y4 * w23
        y5 = sigmoid(z5)

        EXPLAIN['z5'] = z5  
        EXPLAIN['y5'] = y5

        explanation = f"Z5 = Y3*w13 + Y4*w23 = {y3:.4f}*{w13} + {y4:.4f}*{w23} = {z5:.4f}\n" \
                      f"Y5 = sigmoid({z5:.4f}) = {y5:.4f}\n\n" \
                      f"This is the final prediction before and after activation."

        Label(self, text=explanation, font=("Comic Sans MS", 13), bg="#fff0f5", justify=LEFT).pack(padx=20, pady=10)
        Button(self, text="Next → Loss Calculation", font=("Comic Sans MS", 12), bg="#ffb6c1",
               command=lambda: controller.show_frame(LossScreen)).pack(pady=20)


class LossScreen(Frame):
    def __init__(self, parent, controller):
        Frame.__init__(self, parent, bg="#ffe6f0")

        try:
            backprop_img_path = os.path.join("assets", "backprop.png")
            self.backprop_img = ImageTk.PhotoImage(Image.open(backprop_img_path).resize((400, 350)))
            Label(self, image=self.backprop_img, bg="#ffe6f0").pack(pady=10)
        except:
            Label(self, text="(Couldn't load backprop diagram image)", bg="#ffe6f0").pack()

        Label(self, text="💔 Loss Calculation", font=("Georgia", 20), bg="#ffe6f0", fg="#cc3399").pack(pady=10)

        y5, target = EXPLAIN['y5'], 0.5
        loss = (target - y5)
        EXPLAIN['loss'] = loss

        explanation = f"Loss = (Target - Output) = ({target} - {y5:.4f})  = {loss:.6f}\n\n" \
                      f"This is the error loss — the difference between the target and predicted output."

        Label(self, text=explanation, font=("Comic Sans MS", 13), bg="#ffe6f0", justify=LEFT).pack(padx=20, pady=10)
        Button(self, text="Next → Backpropagation", font=("Comic Sans MS", 12), bg="#ffb6c1",
               command=lambda: controller.show_frame(BackpropScreen)).pack(pady=20)

class BackpropScreen(Frame):
    def __init__(self, parent, controller):
        Frame.__init__(self, parent, bg="#fff0f5")

        try:
            backprop_img_path = os.path.join("assets", "backprop.png")
            self.backprop_img = ImageTk.PhotoImage(Image.open(backprop_img_path).resize((400, 350)))
            Label(self, image=self.backprop_img, bg="#fff0f5").pack(pady=10)
        except:
            Label(self, text="(Couldn't load backprop diagram image)", bg="#fff0f5").pack()

        Label(self, text="🔄 Backpropagation Explained", font=("Georgia", 20), bg="#fff0f5", fg="#cc3399").pack(pady=10)

        explanation = "Backpropagation adjusts the weights by calculating the gradient of the loss\nfunction with respect to each weight, using the chain rule.\n\nThese gradients help the model learn which direction to shift the weights\nto minimize future loss."

        Label(self, text=explanation, font=("Comic Sans MS", 13), bg="#fff0f5", justify=LEFT).pack(padx=20, pady=10)
        Button(self, text="Next → Gradient Calculation", font=("Comic Sans MS", 12), bg="#ffb6c1", command=lambda: controller.show_frame(GradientCalcScreen)).pack(pady=20)

class GradientCalcScreen(Frame):
    def __init__(self, parent, controller):
        Frame.__init__(self, parent, bg="#ffe6f0")

        try:
           backprop_img_path = os.path.join("assets", "backprop.png")
           self.backprop_img = ImageTk.PhotoImage(Image.open(backprop_img_path).resize((300, 250)))
           Label(self, image=self.backprop_img, bg="#fff0f5").pack(pady=10)
        except:
            Label(self, text="(Couldn't load backprop diagram image)", bg="#ffe6f0").pack()

        Label(self, text="🧠 Gradient Descent Step (Simplified)", font=("Georgia", 20), bg="#ffe6f0", fg="#cc3399").pack(pady=10)

        # Extract values
        y5 = EXPLAIN['y5']
        z5 = EXPLAIN['z5']
        target = 0.5

        # Store y5 before weight update
        EXPLAIN['y5_before'] = y5  

        # Step 1: Error
        error = y5 - target

        # Step 2: Derivative of loss wrt y5
        dloss_dy5 =  error

        # Step 3: Derivative of sigmoid wrt z5
        dy5_dz5 = sigmoid_derivative(z5)

        # Step 4: Gradients
        y3 = EXPLAIN['y3']
        y4 = EXPLAIN['y4']
        grad_w13 = dloss_dy5 * dy5_dz5 * y3
        grad_w23 = dloss_dy5 * dy5_dz5 * y4

        # Store gradients
        EXPLAIN['grad_w13'] = grad_w13
        EXPLAIN['grad_w23'] = grad_w23

        # Explanation text
        explanation = (
            f"Step-by-step Gradient for Output Layer:\n\n"
            f"🔹 Error = y5 - target = {y5:.4f} - {target} = {error:.4f}\n"
            f"🔹 d(Loss)/dy5 = 2 × Error = {dloss_dy5:.4f}\n"
            f"🔹 dy5/dz5 (Sigmoid Derivative) = {dy5_dz5:.4f}\n\n"
            f"➡️ Gradient w13 = d(Loss) × Sigmoid Derivative × y3 = {grad_w13:.4f}\n"
            f"➡️ Gradient w23 = d(Loss) × Sigmoid Derivative × y4 = {grad_w23:.4f}\n\n"
            "These gradients show how to nudge weights w13 & w23 to reduce the loss."
        )

        Label(self, text=explanation, font=("Comic Sans MS", 13), bg="#ffe6f0", justify=LEFT).pack(padx=20, pady=10)

        Button(self, text="Next → Update Weights", font=("Comic Sans MS", 12), bg="#ffb6c1",
               command=lambda: controller.show_frame(WeightUpdateScreen)).pack(pady=20)

class WeightUpdateScreen(Frame):
    def __init__(self, parent, controller):
        Frame.__init__(self, parent, bg="#fff0f5")

        try:
            backprop_img_path = os.path.join("assets", "backprop.png")
            self.backprop_img = ImageTk.PhotoImage(Image.open(backprop_img_path).resize((400, 350)))
            Label(self, image=self.backprop_img, bg="#fff0f5").pack(pady=10)
        except:
            Label(self, text="(Couldn't load backprop diagram image)", bg="#fff0f5").pack()

        Label(self, text="📉 Weight Update Step", font=("Georgia", 20), bg="#fff0f5", fg="#cc3399").pack(pady=10)

        grad_w13 = EXPLAIN['grad_w13']
        grad_w23 = EXPLAIN['grad_w23']
        y3 = EXPLAIN['y3']
        y4 = EXPLAIN['y4']

        # Original weights
        w13_old = 0.3
        w23_old = 0.9
        lr = 0.1

        w13_new = w13_old - lr * grad_w13
        w23_new = w23_old - lr * grad_w23

        # 💡 Recalculate prediction with updated weights
        z5_new = y3 * w13_new + y4 * w23_new
        y5_new = sigmoid(z5_new)
        EXPLAIN['y5_after'] = y5_new

        explanation = (
            f"Let’s update the weights using:\n"
            f"📌 New Weight = Old Weight - Learning Rate × Gradient\n\n"
            f"w13 = {w13_old} - {lr} × {grad_w13:.4f} = {w13_new:.4f}\n"
            f"w23 = {w23_old} - {lr} × {grad_w23:.4f} = {w23_new:.4f}\n\n"
            f"🔄 Recalculated Output: y5 = sigmoid({z5_new:.4f}) = {y5_new:.4f}\n\n"
            "Now our network has ‘learned’ from the error ✨\n"
            "This process repeats again and again!"
        )

        Label(self, text=explanation, font=("Comic Sans MS", 13), bg="#fff0f5", justify=LEFT).pack(padx=20, pady=10)

        Button(self, text="🎉 Done!Lets's Compare the outputs", font=("Comic Sans MS", 12), bg="#ffb6c1",
               command=lambda: controller.show_frame(OutputComparisonScreen)).pack(pady=20)

class OutputComparisonScreen(Frame):
    def __init__(self, parent, controller):
        Frame.__init__(self, parent, bg="#ffe6f0")

        Label(self, text="🔁 Comparing Outputs: Before vs After", font=("Georgia", 20), bg="#ffe6f0", fg="#cc3399").pack(pady=10)

        # Retrieve previous and new outputs
        y5_before = EXPLAIN.get("y5_before", 0)
        y5_after = EXPLAIN.get("y5_after", 0)

        explanation = (
            f"✨ Before Weight Update:\nOutput = {y5_before:.4f}\n\n"
            f"✨ After Weight Update:\nOutput = {y5_after:.4f}\n\n"
            "We can see that the output has slightly shifted — this means the network\n"
            "is **learning**! The weights got updated, and now the prediction is a bit closer\nto the actual target value.\n\n"
            "Imagine doing this hundreds of times (aka epochs) — the network becomes\nsmarter and more accurate!"
        )

        Label(self, text=explanation, font=("Comic Sans MS", 13), bg="#ffe6f0", justify=LEFT).pack(padx=20, pady=10)
        Button(self, text="Next → Summary", font=("Comic Sans MS", 12), bg="#ffb6c1",
               command=lambda: controller.show_frame(SummaryScreen)).pack(pady=20)
        
class SummaryScreen(Frame):
    def __init__(self, parent, controller):
        Frame.__init__(self, parent, bg="#fff0f5")

        Label(self, text="📚 What Did We Learn?", font=("Georgia", 26, "bold"), bg="#fff0f5", fg="#cc3399").pack(pady=10)

        summary_text = (
            "1. We started with some inputs (x1, x2).\n"
            "2. Fed them into a neural network with hidden neurons.\n"
            "3. Did weighted sums and passed them through a sigmoid activation.\n"
            "4. Got a final prediction (y5).\n"
            "5. Compared it with the actual target to calculate the error.\n"
            "6. Then we used backpropagation to calculate gradients.\n"
            "7. Used those gradients to tweak the weights — making the prediction better.\n\n"
            "And that, my friend, is how learning works in a neural network. \n"
            "One small step of math, one giant leap for machine learning!"
        )

        Label(self, text=summary_text, font=("Comic Sans MS", 20), bg="#fff0f5", justify=LEFT).pack(padx=20, pady=20)

        Button(self, text="🏁 Back to Start", font=("Comic Sans MS", 12), bg="#ffb6c1",
               command=lambda: controller.show_frame(WelcomeScreen)).pack(pady=10)

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
        for F in (WelcomeScreen, DiagramScreen, InputExplanationScreen, HiddenInputScreen, ActivationScreen, OutputScreen, LossScreen, BackpropScreen, GradientCalcScreen,WeightUpdateScreen,OutputComparisonScreen,SummaryScreen):
            frame = F(container, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(WelcomeScreen)

    def show_frame(self, cont):
        self.frames[cont].tkraise()

if __name__ == "__main__":
    app = BrainspellApp()
    app.mainloop()

