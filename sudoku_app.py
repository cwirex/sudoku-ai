"""
Streamlit Web App for Interactive Sudoku Solver
PACKAGE IMPORT VERSION - Run from root directory
Run with: streamlit run sudoku_app.py
"""

import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import os

# Use package imports (since test shows these work perfectly!)
from src.interactive_solver import InteractiveSudokuSolver
from src.utils import setup_project_paths

# Configure matplotlib for web use
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'

# Page configuration
st.set_page_config(
    page_title="Interactive Sudoku Solver",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .status-success {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .status-error {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .hint-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

def init_session_state():
    """Initialize session state variables."""
    defaults = {
        'solver': None,
        'image_processed': False,
        'board_state': 'initial',
        'last_hint': None,
        'error_message': None,
        'processing': False
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

def create_solver():
    """Create and cache the solver instance."""
    if st.session_state.solver is not None:
        return st.session_state.solver, None
    
    try:
        # Use standard models directory
        models_dir = 'models'
        
        solver = InteractiveSudokuSolver(
            confidence_threshold=0.5,
            models_dir=models_dir
        )
        
        st.session_state.solver = solver
        return solver, None
        
    except Exception as e:
        error_msg = f"Failed to initialize solver: {e}\n\nPlease ensure models are trained and available in 'models/' directory."
        return None, error_msg

def process_uploaded_image(uploaded_file):
    """Process uploaded image with proper error handling."""
    try:
        # Read the image
        image = Image.open(uploaded_file)
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array
        image_array = np.array(image)
        
        return image_array, None
        
    except Exception as e:
        return None, f"Error processing image: {e}"

def create_board_figure(solver, title="Current Board", highlight_problematic=False, show_hint=None):
    """Create a matplotlib figure of the board - streamlit compatible."""
    try:
        fig, ax = plt.subplots(figsize=(8, 8))
        fig.patch.set_facecolor('white')
        
        if solver.current_board is None:
            ax.text(0.5, 0.5, 'No board available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            return fig
        
        # Draw the board background
        ax.fill([0, 9, 9, 0], [0, 0, 9, 9], 'white')

        # Draw grid lines
        for i in range(10):
            lw = 2 if i % 3 == 0 else 0.5
            ax.axhline(y=i, color='black', linewidth=lw)
            ax.axvline(x=i, color='black', linewidth=lw)

        # Add numbers with color coding
        for i in range(9):
            for j in range(9):
                if solver.current_board.grid[i, j] != 0:
                    # Determine color
                    color = 'black'  # Default for printed
                    
                    # Check if this is an applied hint
                    if hasattr(solver, 'applied_hints') and (i, j) in solver.applied_hints:
                        color = 'darkgreen'  # Applied hints
                    elif solver.current_board.handwritten_mask[i, j] == 1:
                        if highlight_problematic and hasattr(solver, 'problematic_positions') and (i, j) in solver.problematic_positions:
                            color = 'red'  # Problematic handwritten
                        else:
                            color = 'darkblue'  # Normal handwritten
                    
                    ax.text(j + 0.5, i + 0.5, str(solver.current_board.grid[i, j]),
                           ha='center', va='center',
                           color=color,
                           fontsize=14,
                           fontweight='bold')

        # Show hint if provided
        if show_hint:
            # Highlight the hint position
            ax.add_patch(plt.Rectangle((show_hint.col, show_hint.row), 1, 1, 
                                      fill=False, edgecolor='lime', linewidth=3))
            
            # Show the suggested number
            ax.text(show_hint.col + 0.5, show_hint.row + 0.5, str(show_hint.value),
                   ha='center', va='center',
                   color='lime',
                   fontsize=16,
                   fontweight='bold')

        ax.set_xlim(0, 9)
        ax.set_ylim(9, 0)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('equal')
        ax.axis('off')
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        # Return error figure
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.text(0.5, 0.5, f'Error creating board: {e}', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Error')
        return fig

def create_solution_figure(solver):
    """Create solution visualization with color coding."""
    try:
        if not solver.current_board or not solver.current_board.is_valid:
            return None, "Cannot solve invalid board"
        
        # Create a copy for solving - use package imports
        from src.improved_solver import ImprovedSudokuBoard
        from src.sudoku_solver import SudokuSolver
        
        solve_board = ImprovedSudokuBoard(solver.current_board.grid.copy())
        solve_board.handwritten_mask = solver.current_board.handwritten_mask.copy()
        
        # Solve
        solution_found = SudokuSolver.solve(solve_board)
        
        if not solution_found:
            return None, "Could not find solution"
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 8))
        fig.patch.set_facecolor('white')
        
        ax.fill([0, 9, 9, 0], [0, 0, 9, 9], 'white')

        # Draw grid lines
        for i in range(10):
            lw = 2 if i % 3 == 0 else 0.5
            ax.axhline(y=i, color='black', linewidth=lw)
            ax.axvline(x=i, color='black', linewidth=lw)

        # Add numbers with color coding
        for i in range(9):
            for j in range(9):
                value = solve_board.solution[i, j]
                
                # Determine color based on origin
                if solver.current_board.grid[i, j] != 0:
                    # Original number
                    if solver.current_board.handwritten_mask[i, j] == 1:
                        color = 'darkblue'  # Handwritten
                    else:
                        color = 'black'  # Printed
                else:
                    # Filled by solution
                    color = 'darkgreen'
                
                ax.text(j + 0.5, i + 0.5, str(value),
                       ha='center', va='center',
                       color=color,
                       fontsize=14,
                       fontweight='bold')

        ax.set_xlim(0, 9)
        ax.set_ylim(9, 0)
        ax.set_title("Complete Solution", fontsize=14, fontweight='bold')
        ax.axis('equal')
        ax.axis('off')
        
        plt.tight_layout()
        return fig, None
        
    except Exception as e:
        return None, f"Error creating solution: {e}"

def main():
    # Initialize session state
    init_session_state()
    
    # Header
    st.markdown('<div class="main-header"><h1>🧩 Interactive Sudoku Solver</h1><p>Upload a Sudoku image and solve it step by step with AI assistance!</p></div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🎯 Controls")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload Sudoku Image", 
            type=['png', 'jpg', 'jpeg'],
            help="Upload a clear image of a Sudoku puzzle"
        )
        
        if uploaded_file is not None:
            # Show uploaded image preview
            st.image(uploaded_file, caption="Uploaded Image", width=200)
            
            # Process button
            if st.button("🔍 Process Image", type="primary", disabled=st.session_state.processing):
                st.session_state.processing = True
                
                with st.spinner("Processing image..."):
                    # Create solver
                    solver, error = create_solver()
                    
                    if error:
                        st.session_state.error_message = error
                        st.session_state.processing = False
                        st.rerun()
                    
                    # Process image
                    image_array, img_error = process_uploaded_image(uploaded_file)
                    
                    if img_error:
                        st.session_state.error_message = img_error
                        st.session_state.processing = False
                        st.rerun()
                    
                    # Process with solver
                    try:
                        success = solver.process_image(image_array, display_results=False)
                        
                        if success:
                            st.session_state.image_processed = True
                            st.session_state.board_state = 'processed'
                            st.session_state.error_message = None
                        else:
                            st.session_state.error_message = "Failed to process image. Please try a different image with a clear Sudoku puzzle."
                            
                    except Exception as e:
                        st.session_state.error_message = f"Processing error: {e}"
                    
                    st.session_state.processing = False
                    st.rerun()
        
        # Show controls only if image is processed
        if st.session_state.image_processed and st.session_state.solver is not None:
            st.divider()
            st.header("🎮 Actions")
            
            solver = st.session_state.solver
            board = solver.current_board
            
            if board is not None:
                if board.is_valid:
                    st.markdown('<div class="status-success">✅ Board is Valid!</div>', unsafe_allow_html=True)
                    
                    # Hint button
                    if st.button("💡 Get Hint", key="hint_btn"):
                        try:
                            hint = solver.suggest_hint()
                            if hint:
                                st.session_state.last_hint = hint
                                st.session_state.board_state = 'hint_shown'
                            else:
                                st.session_state.board_state = 'no_hints'
                            st.rerun()
                        except Exception as e:
                            st.session_state.error_message = f"Error getting hint: {e}"
                            st.rerun()
                    
                    # Apply hint button (show only if hint exists)
                    if st.session_state.last_hint is not None:
                        if st.button("✅ Apply Hint & Next", type="primary", key="apply_btn"):
                            try:
                                completed = solver.apply_hint_and_next()
                                st.session_state.last_hint = solver.last_hint  # Update with new hint
                                
                                if solver.current_board.is_complete():
                                    st.session_state.board_state = 'completed'
                                elif st.session_state.last_hint:
                                    st.session_state.board_state = 'hint_shown'
                                else:
                                    st.session_state.board_state = 'no_hints'
                                    
                                st.rerun()
                            except Exception as e:
                                st.session_state.error_message = f"Error applying hint: {e}"
                                st.rerun()
                    
                    # Complete solution button
                    if st.button("🎯 Show Solution", key="solution_btn"):
                        st.session_state.board_state = 'solution_shown'
                        st.rerun()
                    
                else:
                    st.markdown('<div class="status-error">❌ Board is Invalid!</div>', unsafe_allow_html=True)
                    
                    # Show problematic cells button
                    if st.button("🔍 Show Problems", key="problems_btn"):
                        try:
                            solver._identify_problematic_cells()
                            st.session_state.board_state = 'problems_shown'
                            st.rerun()
                        except Exception as e:
                            st.session_state.error_message = f"Error identifying problems: {e}"
                            st.rerun()
                    
                    # Manual correction section
                    st.subheader("✏️ Manual Correction")
                    with st.form("correction_form"):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            row = st.number_input("Row", min_value=1, max_value=9, value=1)
                        with col2:
                            col_num = st.number_input("Col", min_value=1, max_value=9, value=1)
                        with col3:
                            value = st.number_input("Value", min_value=0, max_value=9, value=0)
                        
                        if st.form_submit_button("Apply Correction"):
                            try:
                                # Convert to 0-based indexing
                                row_idx = row - 1
                                col_idx = col_num - 1
                                
                                # Apply correction
                                old_value = solver.current_board.grid[row_idx, col_idx]
                                solver.current_board.grid[row_idx, col_idx] = value
                                
                                # Re-validate
                                is_valid = solver.current_board.validate()
                                solver.current_board.is_valid = is_valid
                                
                                st.session_state.board_state = 'corrected'
                                st.success(f"Applied: ({row},{col_num}) {old_value} → {value}")
                                
                                if is_valid:
                                    st.success("Board is now valid!")
                                
                                st.rerun()
                                
                            except Exception as e:
                                st.session_state.error_message = f"Error applying correction: {e}"
                                st.rerun()
            
            # Reset button
            st.divider()
            if st.button("🔄 Reset", key="reset_btn"):
                # Clear all session state
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
    
    # Main content area
    if st.session_state.error_message:
        st.error(f"❌ {st.session_state.error_message}")
        st.session_state.error_message = None
    
    if not st.session_state.image_processed:
        # Welcome screen
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ## 🎯 How to Use
            
            1. **Upload** a clear image of a Sudoku puzzle using the sidebar
            2. **Process** the image to detect and extract the board
            3. **Interact** with the puzzle:
               - Get intelligent hints
               - Apply hints step by step
               - Fix any recognition errors
               - View the complete solution
            
            ## 📋 Features
            
            - 🤖 **AI-powered digit recognition** (printed & handwritten)
            - 🎯 **Smart board detection** and perspective correction
            - 💡 **Intelligent hints** with difficulty levels
            - ✏️ **Manual correction** for any errors
            - 🎨 **Color-coded visualization**
            - 🧩 **Step-by-step solving** guidance
            """)
        
        with col2:
            st.markdown("## 🎨 Color Guide")
            st.markdown("""
            - **Black**: Printed digits
            - **Dark Blue**: Handwritten digits  
            - **Dark Green**: Applied hints
            - **Red**: Problematic cells
            - **Lime**: Current hint
            """)
    
    else:
        # Show board and interactions
        solver = st.session_state.solver
        
        if solver and solver.current_board:
            # Main board display
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Board visualization based on state
                if st.session_state.board_state == 'hint_shown' and st.session_state.last_hint:
                    st.subheader("💡 Hint Suggestion")
                    
                    # Create hint visualization
                    fig = create_board_figure(
                        solver, 
                        f"Hint: Place {st.session_state.last_hint.value} at position ({st.session_state.last_hint.row+1}, {st.session_state.last_hint.col+1})",
                        show_hint=st.session_state.last_hint
                    )
                    st.pyplot(fig)
                    plt.close(fig)
                    
                    # Show hint details
                    hint = st.session_state.last_hint
                    st.markdown(f"""
                    <div class="hint-box">
                    <strong>{hint.difficulty.upper()} HINT:</strong><br>
                    Place <strong>{hint.value}</strong> at row <strong>{hint.row+1}</strong>, column <strong>{hint.col+1}</strong><br>
                    <em>{hint.reason}</em>
                    </div>
                    """, unsafe_allow_html=True)
                
                elif st.session_state.board_state == 'solution_shown':
                    st.subheader("🎯 Complete Solution")
                    
                    solution_fig, error = create_solution_figure(solver)
                    if solution_fig:
                        st.pyplot(solution_fig)
                        plt.close(solution_fig)
                        
                        # Show solution statistics
                        original_filled = np.sum(solver.current_board.grid != 0)
                        st.success(f"✅ Solution found! Added {81 - original_filled} numbers to complete the puzzle.")
                    else:
                        st.error(f"❌ {error}")
                
                elif st.session_state.board_state == 'problems_shown':
                    st.subheader("🔍 Problematic Cells")
                    
                    fig = create_board_figure(solver, "Problematic Handwritten Cells", highlight_problematic=True)
                    st.pyplot(fig)
                    plt.close(fig)
                    
                    # List problematic positions
                    if hasattr(solver, 'problematic_positions') and solver.problematic_positions:
                        st.warning(f"Found {len(solver.problematic_positions)} problematic handwritten cells:")
                        for i, (row, col) in enumerate(solver.problematic_positions, 1):
                            value = solver.current_board.grid[row, col]
                            st.write(f"{i}. Position ({row+1}, {col+1}): **{value}**")
                    else:
                        st.info("No specific problematic cells identified.")
                
                elif st.session_state.board_state == 'completed':
                    st.success("🎉 Congratulations! Puzzle Completed!")
                    
                    fig = create_board_figure(solver, "Completed Puzzle!")
                    st.pyplot(fig)
                    plt.close(fig)
                    
                    st.balloons()  # Celebration effect!
                
                elif st.session_state.board_state == 'no_hints':
                    st.subheader("🧩 Current Board")
                    
                    fig = create_board_figure(solver, "Current Board - No More Hints Available")
                    st.pyplot(fig)
                    plt.close(fig)
                    
                    st.info("No more hints available. The puzzle might be nearly complete or require advanced solving techniques.")
                
                else:
                    # Default board view
                    st.subheader("🧩 Current Board")
                    
                    fig = create_board_figure(solver, "Digital Board (After Auto-Correction)")
                    st.pyplot(fig)
                    plt.close(fig)
            
            with col2:
                # Board statistics
                st.subheader("📊 Board Status")
                
                stats = solver.current_board.get_stats()
                
                # Status indicator
                if stats['is_valid']:
                    st.success("✅ Valid")
                else:
                    st.error("❌ Invalid")
                
                # Metrics
                st.metric("Filled", f"{stats['filled_cells']}/81")
                st.metric("Progress", stats['fill_percentage'])
                st.metric("Mode", stats['mode'].title())
                
                # Cell breakdown
                st.subheader("📈 Breakdown")
                st.write(f"**Printed:** {stats['printed_cells']}")
                st.write(f"**Handwritten:** {stats['handwritten_cells']}")
                st.write(f"**Empty:** {stats['empty_cells']}")
                
                if hasattr(solver, 'applied_hints') and solver.applied_hints:
                    st.write(f"**Applied Hints:** {len(solver.applied_hints)}")

if __name__ == "__main__":
    main()