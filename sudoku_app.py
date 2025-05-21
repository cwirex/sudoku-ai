"""
Streamlit Web App for Interactive Sudoku Solver - RIGHT PANEL VERSION
PACKAGE IMPORT VERSION - Run from root directory
Run with: streamlit run sudoku_app.py
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from src.processing.interactive_solver import InteractiveSudokuSolver

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

# Enhanced CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        color: white;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
    }
    .main-header h1 {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .main-header p {
        font-size: 1.2rem;
        opacity: 0.9;
    }
    .minimal-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .minimal-header h1 {
        font-size: 1.8rem;
        margin: 0;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }
    .status-success {
        background: linear-gradient(135deg, #d4edda, #c3e6cb);
        border: 2px solid #28a745;
        color: #155724;
        padding: 15px;
        border-radius: 10px;
        margin: 15px 0;
        box-shadow: 0 4px 8px rgba(40, 167, 69, 0.2);
        font-weight: bold;
    }
    .status-error {
        background: linear-gradient(135deg, #f8d7da, #f5c6cb);
        border: 2px solid #dc3545;
        color: #721c24;
        padding: 15px;
        border-radius: 10px;
        margin: 15px 0;
        box-shadow: 0 4px 8px rgba(220, 53, 69, 0.2);
        font-weight: bold;
    }
    .hint-box {
        background: linear-gradient(135deg, #fff3cd, #ffeaa7);
        border: 2px solid #ffc107;
        color: #856404;
        padding: 20px;
        border-radius: 15px;
        margin: 15px 0;
        box-shadow: 0 6px 12px rgba(255, 193, 7, 0.2);
        font-size: 1.1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
        border: 1px solid #dee2e6;
        border-radius: 10px;
        padding: 12px;
        margin: 8px 0;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card h4 {
        color: #2c3e50;
        margin: 0 0 5px 0;
        font-size: 0.9rem;
    }
    .metric-card h3 {
        margin: 0;
        font-size: 1.3rem;
    }
    .metric-card p {
        margin: 0;
        font-size: 0.8rem;
        color: #666;
    }
    .stButton > button {
        width: 100%;
        border-radius: 10px;
        border: none;
        padding: 0.75rem 1rem;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .upload-section {
        border: 2px dashed #ccc;
        border-radius: 15px;
        padding: 15px;
        background: linear-gradient(135deg, #f8f9fa, #ffffff);
        margin-bottom: 15px;
    }
    .control-section {
        background: linear-gradient(135deg, #ffffff, #f8f9fa);
        border: 1px solid #dee2e6;
        border-radius: 15px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    .sidebar-header {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 8px;
        border-radius: 8px;
        text-align: center;
        margin-bottom: 10px;
        font-weight: bold;
        font-size: 0.9rem;
    }
    .panel-header {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 10px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 15px;
        font-weight: bold;
        font-size: 1rem;
    }
    .stats-section {
        background: linear-gradient(135deg, #f0f8ff, #e6f3ff);
        border: 1px solid #b3d9ff;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    .right-panel {
        background: linear-gradient(135deg, #fafafa, #ffffff);
        border: 1px solid #e0e0e0;
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
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
    """Create a matplotlib figure of the board with improved visibility."""
    try:
        fig, ax = plt.subplots(figsize=(12, 12))  # Larger figure for better visibility
        fig.patch.set_facecolor('white')
        
        if solver.current_board is None:
            ax.text(0.5, 0.5, 'No board available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            return fig
        
        # Draw the board background with subtle color
        ax.fill([0, 9, 9, 0], [0, 0, 9, 9], '#fafafa')

        # Draw grid lines with enhanced visibility
        for i in range(10):
            if i % 3 == 0:
                # Thick lines for 3x3 boxes
                ax.axhline(y=i, color='#2c3e50', linewidth=3)
                ax.axvline(x=i, color='#2c3e50', linewidth=3)
            else:
                # Thin lines for cells
                ax.axhline(y=i, color='#7f8c8d', linewidth=1)
                ax.axvline(x=i, color='#7f8c8d', linewidth=1)

        # Add numbers with improved colors and size
        for i in range(9):
            for j in range(9):
                if solver.current_board.grid[i, j] != 0:
                    # Determine color with better contrast
                    color = '#2c3e50'  # Default dark blue-gray for printed
                    
                    # Check if this is an applied hint
                    if hasattr(solver, 'applied_hints') and (i, j) in solver.applied_hints:
                        color = '#27ae60'  # Bright green for applied hints
                    elif solver.current_board.handwritten_mask[i, j] == 1:
                        if highlight_problematic and hasattr(solver, 'problematic_positions') and (i, j) in solver.problematic_positions:
                            color = '#e74c3c'  # Bright red for problematic handwritten
                        else:
                            color = '#3498db'  # Bright blue for normal handwritten
                    
                    ax.text(j + 0.5, i + 0.5, str(solver.current_board.grid[i, j]),
                           ha='center', va='center',
                           color=color,
                           fontsize=20,  # Increased from 18 to 20
                           fontweight='bold')

        # Show hint if provided with unified background styling
        if show_hint:
            # Unified background highlight for hint position - solid, clean look
            from matplotlib.patches import Rectangle
            hint_rect = Rectangle((show_hint.col, show_hint.row), 1, 1, 
                                fill=True, facecolor='#fff3cd', alpha=1.0)
            ax.add_patch(hint_rect)
            
            # Show the suggested number with unified background styling
            ax.text(show_hint.col + 0.5, show_hint.row + 0.5, str(show_hint.value),
                   ha='center', va='center',
                   color='#2c3e50',  # Normal color for hint number
                   fontsize=20,  # Same size as other digits
                   fontweight='bold')

        ax.set_xlim(0, 9)
        ax.set_ylim(9, 0)
        ax.set_title(title, fontsize=18, fontweight='bold', pad=20)
        ax.axis('equal')
        ax.axis('off')
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        # Return error figure
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.text(0.5, 0.5, f'Error creating board: {e}', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Error')
        return fig

def create_solution_figure(solver):
    """Create solution visualization with enhanced colors."""
    try:
        if not solver.current_board or not solver.current_board.is_valid:
            return None, "Cannot solve invalid board"
        
        # Create a copy for solving - use package imports
        from src.processing.improved_solver import ImprovedSudokuBoard
        from src.processing.solver import SudokuSolver
        
        solve_board = ImprovedSudokuBoard(solver.current_board.grid.copy())
        solve_board.handwritten_mask = solver.current_board.handwritten_mask.copy()
        
        # Solve
        solution_found = SudokuSolver.solve(solve_board)
        
        if not solution_found:
            return None, "Could not find solution"
        
        # Create figure with enhanced styling
        fig, ax = plt.subplots(figsize=(12, 12))
        fig.patch.set_facecolor('white')
        
        ax.fill([0, 9, 9, 0], [0, 0, 9, 9], '#f8fff8')  # Very light green background

        # Draw grid lines
        for i in range(10):
            if i % 3 == 0:
                ax.axhline(y=i, color='#2c3e50', linewidth=3)
                ax.axvline(x=i, color='#2c3e50', linewidth=3)
            else:
                ax.axhline(y=i, color='#7f8c8d', linewidth=1)
                ax.axvline(x=i, color='#7f8c8d', linewidth=1)

        # Add numbers with enhanced color coding
        for i in range(9):
            for j in range(9):
                value = solve_board.solution[i, j]
                
                # Determine color based on origin with better contrast
                if solver.current_board.grid[i, j] != 0:
                    # Original number
                    if solver.current_board.handwritten_mask[i, j] == 1:
                        color = '#3498db'  # Bright blue for handwritten
                    else:
                        color = '#2c3e50'  # Dark blue-gray for printed
                    bbox_color = 'white'
                else:
                    # Filled by solution
                    color = '#27ae60'  # Bright green for solution numbers
                    bbox_color = '#e8f5e8'  # Light green background
                
                ax.text(j + 0.5, i + 0.5, str(value),
                       ha='center', va='center',
                       color=color,
                       fontsize=20,  # Increased size
                       fontweight='bold')

        ax.set_xlim(0, 9)
        ax.set_ylim(9, 0)
        ax.set_title("🎯 Complete Solution", fontsize=18, fontweight='bold', pad=20)
        ax.axis('equal')
        ax.axis('off')
        
        plt.tight_layout()
        return fig, None
        
    except Exception as e:
        return None, f"Error creating solution: {e}"

def main():
    # Initialize session state
    init_session_state()
    
    # Conditional Header - full on welcome, minimal after processing
    if not st.session_state.image_processed:
        st.markdown('''
        <div class="main-header">
            <h1>🧩 Interactive Sudoku Solver</h1>
            <p>Upload a Sudoku image and solve it step by step with AI assistance!</p>
        </div>
        ''', unsafe_allow_html=True)
    else:
        st.markdown('''
        <div class="minimal-header">
            <h1>🧩 Interactive Sudoku Solver</h1>
        </div>
        ''', unsafe_allow_html=True)
    
    # Left Sidebar - ONLY upload and preview
    with st.sidebar:
        # Upload Section
        st.markdown('<div class="sidebar-header">📁 Upload Image</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Choose a Sudoku image", 
            type=['png', 'jpg', 'jpeg'],
            help="Upload a clear image of a Sudoku puzzle",
            label_visibility="collapsed"
        )
        
        # Features section (only show when no image uploaded)
        if uploaded_file is None:
            st.markdown('<div class="sidebar-header">✨ Features</div>', unsafe_allow_html=True)
            st.markdown('''
            - 🤖 **AI-powered digit recognition** (printed & handwritten)
            - 🎯 **Smart board detection** and perspective correction
            - 💡 **Intelligent hints** with difficulty levels
            - ✏️ **Manual correction** for any errors
            - 🎨 **Color-coded visualization**
            - 🧩 **Step-by-step solving** guidance
            ''')
        
        
        # Image Preview Section (only show when file uploaded)
        if uploaded_file is not None:
            st.markdown('<div class="sidebar-header">📷 Image Preview</div>', unsafe_allow_html=True)
            st.image(uploaded_file, caption="Uploaded Image", width=280)
            
            # Process button
            if st.button("🔍 Process Image", type="primary", disabled=st.session_state.processing, key="process_btn"):
                st.session_state.processing = True
                
                with st.spinner("🔄 Processing image..."):
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
                            st.success("✅ Image processed successfully!")
                        else:
                            st.session_state.error_message = "Failed to process image. Please try a different image with a clear Sudoku puzzle."
                            
                    except Exception as e:
                        st.session_state.error_message = f"Processing error: {e}"
                    
                    st.session_state.processing = False
                    st.rerun()
        
        # Reset button (only show if image processed)
        if st.session_state.image_processed:
            st.markdown("---")
            if st.button("🔄 Reset All", key="reset_btn"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
    
    # Main content area
    if st.session_state.error_message:
        st.error(f"❌ {st.session_state.error_message}")
        st.session_state.error_message = None
    
    if not st.session_state.image_processed:
        # Enhanced welcome screen with rich content
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("""
            ## 🎯 How to Use
            
            ### Step 1: Upload Image 📁
            Use the sidebar to upload a clear image of a Sudoku puzzle. Supported formats: PNG, JPG, JPEG.
            
            ### Step 2: Process Image 🔍
            Click "Process Image" to detect and extract the Sudoku board using AI.
            
            ### Step 3: Interact with Puzzle 🎮
            - **Get Hints:** Receive intelligent solving suggestions
            - **Apply Hints:** Follow step-by-step guidance
            - **Fix Errors:** Manually correct any recognition mistakes
            - **View Solution:** See the complete solved puzzle
            
            ### Step 4: Solve! 🎉
            Work through the puzzle with AI assistance or jump straight to the solution.
            """)
        
        with col2:
            st.markdown("""
            ## 📝 Tips for Best Results
            
            ✅ **Good Images:**
            - Clear, well-lit photos
            - Straight-on angle
            - High contrast
            - Minimal shadows
            
            ⚠️ **Avoid:**
            - Blurry or tilted images
            - Poor lighting
            - Reflections or glare
            - Partially visible grids
            """)
    
    else:
        # Board display with right panel for actions and stats
        solver = st.session_state.solver
        
        if solver and solver.current_board:
            # Create layout: Board (left) + Controls (right)
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Board visualization based on state
                if st.session_state.board_state == 'hint_shown' and st.session_state.last_hint:
                    # Create hint visualization
                    fig = create_board_figure(
                        solver, 
                        f"💡 Hint: Place {st.session_state.last_hint.value} at position ({st.session_state.last_hint.row+1}, {st.session_state.last_hint.col+1})",
                        show_hint=st.session_state.last_hint
                    )
                    st.pyplot(fig)
                    plt.close(fig)
                    
                    # Show minimal hint details below the board
                    hint = st.session_state.last_hint
                    st.markdown(f"""
                    **Hint:** Place **{hint.value}** at position **({hint.row+1}, {hint.col+1})**
                    """, unsafe_allow_html=True)
                
                elif st.session_state.board_state == 'solution_shown':
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
                    fig = create_board_figure(solver, "🔍 Problematic Handwritten Cells", highlight_problematic=True)
                    st.pyplot(fig)
                    plt.close(fig)
                    
                    # List problematic positions
                    if hasattr(solver, 'problematic_positions') and solver.problematic_positions:
                        st.warning(f"Found {len(solver.problematic_positions)} problematic handwritten cells:")
                        for i, (row, col) in enumerate(solver.problematic_positions, 1):
                            value = solver.current_board.grid[row, col]
                            st.markdown(f"**{i}.** Position ({row+1}, {col+1}): **{value}**")
                    else:
                        st.info("No specific problematic cells identified.")
                
                elif st.session_state.board_state == 'completed':
                    st.success("🎉 Congratulations! Puzzle Completed!")
                    
                    fig = create_board_figure(solver, "🎉 Completed Puzzle!")
                    st.pyplot(fig)
                    plt.close(fig)
                    
                    st.balloons()  # Celebration effect!
                
                elif st.session_state.board_state == 'no_hints':
                    fig = create_board_figure(solver, "🧩 Current Board - No More Hints Available")
                    st.pyplot(fig)
                    plt.close(fig)
                    
                    st.info("🎯 No more hints available. The puzzle might be nearly complete or require advanced solving techniques.")
                
                else:
                    # Default board view
                    fig = create_board_figure(solver, "🧩 Current Sudoku Board")
                    st.pyplot(fig)
                    plt.close(fig)
            
            with col2:
                # RIGHT PANEL: Actions and Statistics
                
                # Actions Section
                st.markdown('<div class="panel-header">🎮 Actions</div>', unsafe_allow_html=True)
                
                board = solver.current_board
                
                if board.is_valid:
                    # Action buttons
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
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
                    
                    with col_b:
                        if st.button("🎯 Solution", key="solution_btn"):
                            st.session_state.board_state = 'solution_shown'
                            st.rerun()
                    
                    # Apply hint button (full width if hint exists)
                    if st.session_state.last_hint is not None:
                        if st.button("✅ Apply Hint & Get Next", type="primary", key="apply_btn"):
                            try:
                                completed = solver.apply_hint_and_next()
                                st.session_state.last_hint = solver.last_hint
                                
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
                    
                else:
                    if st.button("🔍 Show Problems", key="problems_btn"):
                        try:
                            solver._identify_problematic_cells()
                            st.session_state.board_state = 'problems_shown'
                            st.rerun()
                        except Exception as e:
                            st.session_state.error_message = f"Error identifying problems: {e}"
                            st.rerun()
                    
                    # Manual correction section
                    st.markdown("**✏️ Manual Correction**")
                    with st.form("correction_form"):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            row = st.number_input("Row", min_value=1, max_value=9, value=1)
                        with col2:
                            col_num = st.number_input("Col", min_value=1, max_value=9, value=1)
                        with col3:
                            value = st.number_input("Value", min_value=0, max_value=9, value=0)
                        
                        if st.form_submit_button("Apply", type="primary"):
                            try:
                                row_idx = row - 1
                                col_idx = col_num - 1
                                
                                old_value = solver.current_board.grid[row_idx, col_idx]
                                solver.current_board.grid[row_idx, col_idx] = value
                                
                                is_valid = solver.current_board.validate()
                                solver.current_board.is_valid = is_valid
                                
                                st.session_state.board_state = 'corrected'
                                st.success(f"({row},{col_num}) {old_value}→{value}")
                                
                                if is_valid:
                                    st.success("Board is now valid!")
                                
                                st.rerun()
                                
                            except Exception as e:
                                st.session_state.error_message = f"Error: {e}"
                                st.rerun()
                
                # Statistics Section
                st.markdown('<div class="panel-header">📊 Statistics</div>', unsafe_allow_html=True)
                
                stats = solver.current_board.get_stats()
                
                # Progress bar
                progress = stats['filled_cells'] / 81
                st.progress(progress, text=f"Progress: {stats['fill_percentage']}")
                
                # Compact metric cards
                col1, col2 = st.columns(2)
                
                with col1:
                    # Board validity
                    if stats['is_valid']:
                        st.markdown(f'''
                        <div class="metric-card" style="border-color: #28a745;">
                            <h4>✅ Status</h4>
                            <h3 style="color: #27ae60;">Valid</h3>
                        </div>
                        ''', unsafe_allow_html=True)
                    else:
                        st.markdown(f'''
                        <div class="metric-card" style="border-color: #dc3545;">
                            <h4>❌ Status</h4>
                            <h3 style="color: #e74c3c;">Invalid</h3>
                        </div>
                        ''', unsafe_allow_html=True)
                
                with col2:
                    # Handwritten/Printed info
                    if stats['handwritten_cells'] > 0 and stats['printed_cells'] > 0:
                        cell_info = "Mixed"
                        color = "#f39c12"
                    elif stats['handwritten_cells'] > 0:
                        cell_info = "Handwritten"
                        color = "#3498db"
                    elif stats['printed_cells'] > 0:
                        cell_info = "Printed"
                        color = "#2c3e50"
                    else:
                        cell_info = "Empty"
                        color = "#95a5a6"
                        
                    st.markdown(f'''
                    <div class="metric-card">
                        <h4>📝 Type</h4>
                        <h3 style="color: {color};">{cell_info}</h3>
                    </div>
                    ''', unsafe_allow_html=True)
                
                # Breakdown and hints as info cards
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f'''
                    <div class="metric-card">
                        <h4>📊 Breakdown</h4>
                        <p style="font-size: 0.85rem; line-height: 1.4;">
                        Printed: {stats['printed_cells']}<br>
                        Handwritten: {stats['handwritten_cells']}<br>
                        Empty: {stats['empty_cells']}
                        </p>
                    </div>
                    ''', unsafe_allow_html=True)
                
                with col2:
                    hints_count = len(solver.applied_hints) if hasattr(solver, 'applied_hints') and solver.applied_hints else 0
                    st.markdown(f'''
                    <div class="metric-card">
                        <h4>💡 Applied Hints</h4>
                        <h3 style="color: #27ae60;">{hints_count}</h3>
                        <p>hints used</p>
                    </div>
                    ''', unsafe_allow_html=True)

if __name__ == "__main__":
    main()