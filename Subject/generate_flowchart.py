import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_flowchart():
    # Set the figure size and create an axis
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')

    # Define colors
    color_data = '#fae3e3'     # Light red for data
    color_process = '#e3f2fd'  # Light blue for notebooks/processes
    color_insight = '#fff9c4'  # Light yellow for insights
    color_app = '#e8f5e9'      # Light green for app
    edge_col = '#333333'

    # Define nodes: (Text, coordinates(x, y), width, height, color)
    nodes = {
        "Raw":    ("Raw Data CSV\n(heart_disease_data.csv)", (1, 5), 3, 1.5, color_data),
        "P1":     ("Phase 1: Data Preprocessing\nNotebook", (5, 5), 3.3, 1.5, color_process),
        "Clean":  ("Clean CSV\n(cleaned_data.csv)", (9, 5), 2.5, 1.5, color_data),
        "P2":     ("Phase 2: EDA\nNotebook", (9, 8), 2.5, 1.5, color_process),
        "Ins":    ("Insights &\nVisualizations", (5, 8), 2.5, 1.5, color_insight),
        "P3":     ("Phase 3: Model Training\nNotebook", (13, 5), 3.3, 1.5, color_process),
        "Art":    ("Saved Model & Scaler\n(.pkl)", (13, 2), 3, 1.5, color_data),
        "App":    ("Phase 4: Streamlit\nWeb App", (9, 2), 3, 1.5, color_app)
    }

    # Draw boxes and texts
    for k, (text, (x, y), w, h, col) in nodes.items():
        rect = patches.FancyBboxPatch((x - w/2, y - h/2), w, h, 
                                      boxstyle="round,pad=0.2", 
                                      ec=edge_col, fc=col, lw=2)
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', fontsize=11, fontweight='bold', color='#111111')

    # Helper function to draw arrows
    def connect(n1, n2, style="-|>", arrow_color='#333', conn_style="arc3,rad=0"):
        x1, y1 = nodes[n1][1]
        x2, y2 = nodes[n2][1]
        
        # Adjust connection points strictly based on center-to-center distances
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle=style, color=arrow_color, lw=2.5, 
                                    shrinkA=45, shrinkB=45, connectionstyle=conn_style))

    # Connect Nodes
    connect("Raw", "P1")
    connect("P1", "Clean")
    
    connect("Clean", "P2", style="-|>", conn_style="arc3,rad=0.2")
    connect("P2", "Ins", style="->", arrow_color='#f57f17', conn_style="arc3,rad=0")
    
    connect("Clean", "P3")
    connect("Ins", "P3", style="->", arrow_color='#f57f17', conn_style="arc3,rad=-0.1")
    
    connect("P3", "Art")
    connect("Art", "App", style="-|>", conn_style="arc3,rad=0")

    # Final Title and Save
    plt.title("Heart Disease Prediction Project - 4 Phase Pipeline", fontsize=18, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig("pipeline_flowchart.png", dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    draw_flowchart()
    print("Flowchart image generated successfully as 'pipeline_flowchart.png'")
