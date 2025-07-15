import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import plotly.express as px
import plotly.graph_objects as go
import time
import warnings
warnings.filterwarnings('ignore')

# Configuration Parameters - Add at top of file after imports
class OptimizationConfig:
    """Centralized configuration for optimization parameters"""
    
    # Penalty parameters
    CONSTRAINT_VIOLATION_PENALTY = 1e6  # Reduced from 1e9 for stability
    RESTRICTION_VIOLATION_PENALTY = 1e7
    CONSTRAINT_TOLERANCE = 0.01  # Allow small constraint violations
    
    # Threshold parameters
    WEIGHT_SIGNIFICANCE_THRESHOLD = 0.005  # Unified threshold
    RESTRICTED_COMPOUND_LIMIT = 5.0  # Percentage
    PROPORTION_TOLERANCE = 0.05  # ±5% for lot optimization
    
    # Optimization parameters
    MAX_ITERATIONS = 1000
    CONVERGENCE_TOLERANCE = 1e-8
    DEFAULT_N_STARTS = 5
    
    # Scoring parameters
    COST_NORMALIZATION_METHOD = 'minmax'  # 'minmax' or 'zscore'
    PROPERTY_NORMALIZATION_METHOD = 'std'  # 'std' or 'minmax'


# Page configuration
st.set_page_config(
    page_title="Robust Blend Optimizer",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🧪 Robust Blend Optimizer")
st.markdown("Reliable chemical compound blend optimization with comprehensive error handling")

# Initialize session state for workflow
if 'optimization_stage' not in st.session_state:
    st.session_state.optimization_stage = 'initial'
if 'top_blends' not in st.session_state:
    st.session_state.top_blends = []
if 'selected_blend' not in st.session_state:
    st.session_state.selected_blend = None

# Cache data loading with enhanced error handling
@st.cache_data
def load_csv():
    try:
        compound_data = pd.read_csv("compound_dataset_2000.csv")
        
        # Validate required columns
        required_props = [
            'Attrition resistance', 'Thermal Stability', 'Average particle size',
            'Particle size distribution', 'density', 'rare earth oxides',
            'catalyst surface area', 'micropore surface area', 'zeolite surface area',
            'X-ray fluorescence'
        ]
        
        # Check for missing properties
        missing_props = [prop for prop in required_props if prop not in compound_data.columns]
        if missing_props:
            st.error(f"Missing properties in compound data: {missing_props}")
            return None, None
            
        # Generate cost column if not present
        if 'cost' not in compound_data.columns:
            np.random.seed(42)
            compound_data['cost'] = np.random.randint(100, 1000, len(compound_data))
        
        # Load lot data with similar validation
        try:
            lot_compound = pd.read_csv("compound_lots_dataset_2000.csv")
            
            # Ensure lot data has required columns
            if 'cost' not in lot_compound.columns:
                np.random.seed(42)
                lot_compound['cost'] = np.random.randint(100, 1000, len(lot_compound))
                
            return compound_data, lot_compound
        except Exception as e:
            st.warning(f"Lot data not available: {str(e)}")
            return compound_data, None
            
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None

# Load data
compound_data, lot_data = load_csv()

if compound_data is None:
    st.stop()

# Properties list
properties = [
    'Attrition resistance', 'Thermal Stability', 'Average particle size',
    'Particle size distribution', 'density', 'rare earth oxides',
    'catalyst surface area', 'micropore surface area', 'zeolite surface area',
    'X-ray fluorescence'
]

# Restricted compounds (REO-containing compounds)
RESTRICTED_LIMIT = 5.0  # 5% limit for restricted compounds
REO_PROPERTY = 'rare earth oxides'

# Sidebar optimization settings
st.sidebar.header("⚙️ Configuration")

# Cost-accuracy tradeoff
lambda_val = st.sidebar.slider(
    "λ (Cost vs Accuracy Tradeoff)", 
    0.1, 50.0, 1.0, 0.1,
    help="Higher values prioritize cost reduction"
)

top_n = st.sidebar.slider("Top N candidates", 1, 500, 150)

# Pre-filtering settings
st.sidebar.subheader("🔍 Pre-filtering Settings")
cost_weight_prefilter = st.sidebar.slider(
    "Cost Weight in Pre-filtering", 
    0.0, 0.8, 0.3, 0.1,
    help="Higher values prioritize cost in pre-filtering"
)

# Property weights for pre-filtering
st.sidebar.subheader("⚖️ Property Importance")
property_weights = {}
for prop in properties:
    if prop == 'rare earth oxides':
        default_weight = 3.0  # High importance for restrictions
    elif 'cost' in prop.lower():
        default_weight = 2.0  # Cost-related properties
    else:
        default_weight = 1.0
    
    property_weights[prop] = st.sidebar.slider(
        f"{prop[:20]}...", 
        0.1, 5.0, default_weight, 0.1,
        key=f"weight_{prop}"
    )

# Constraint management
st.sidebar.subheader("🎯 Constraint Management")
exclude_compounds = st.sidebar.multiselect(
    "Exclude Compounds", 
    compound_data['Compound Name'].tolist(),
    help="Select compounds to exclude from optimization"
)

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🎯 Target Properties")
    
    # Target property input
    target_dict = {}
    default_values = [87.49, 885.21, 7.59, 1.7, 0.909, 0.78, 64.52, 87.96, 340.45, 0.7081]
    
    for i, prop in enumerate(properties):
        constraint_type = st.selectbox(
            f"Constraint for {prop}:", 
            options=["=", ">=", "<="], 
            key=f"constraint_{i}",
            help="= for exact match, >= for minimum, <= for maximum"
        )
        value = st.number_input(
            f"Target value for {prop}", 
            value=default_values[i], 
            key=f"value_{i}",
            format="%.4f"
        )
        target_dict[prop] = {"type": constraint_type, "value": value}

with col2:
    st.subheader("📊 Target Summary")
    target_df = pd.DataFrame.from_dict(target_dict, orient='index')
    target_df.columns = ['Constraint', 'Target Value']
    st.dataframe(target_df, use_container_width=True)

def blend_error(weights, prop_matrix, constraint_dict, cost_vector, lam=1.0, 
                property_stds=None, reo_values=None, check_restrictions=True, 
                config=None):
    """Enhanced blend error function with configurable penalties and tolerances"""
    
    if config is None:
        config = OptimizationConfig()
    
    # Validate weights with tolerance
    if np.any(weights < -config.CONSTRAINT_TOLERANCE) or abs(np.sum(weights) - 1) > config.CONSTRAINT_TOLERANCE:
        return config.CONSTRAINT_VIOLATION_PENALTY
    
    # Check for restricted compound violation with configurable penalty
    if check_restrictions and reo_values is not None:
        restricted_weight = 0
        for i, weight in enumerate(weights):
            if weight > config.WEIGHT_SIGNIFICANCE_THRESHOLD and reo_values[i] > 0:
                restricted_weight += weight
        
        if restricted_weight > config.RESTRICTED_COMPOUND_LIMIT / 100:
            return config.RESTRICTION_VIOLATION_PENALTY
    
    blend = np.dot(weights, prop_matrix)
    error = 0
    
    # Enhanced constraint handling with tolerance
    for i, prop in enumerate(properties):
        if prop not in constraint_dict:
            continue
            
        val = blend[i]
        rule = constraint_dict[prop]["type"]
        target = constraint_dict[prop]["value"]
        
        # Improved normalization
        if config.PROPERTY_NORMALIZATION_METHOD == 'std':
            std = property_stds[i] if property_stds is not None and property_stds[i] > 0 else 1.0
            normalizer = std
        else:  # minmax normalization
            normalizer = abs(target) if target != 0 else 1.0
        
        # Constraint evaluation with tolerance
        if rule == "=":
            deviation = abs(val - target)
            if deviation > config.CONSTRAINT_TOLERANCE * abs(target):
                error += (deviation / normalizer) ** 2
        elif rule == "<=" and val > target + config.CONSTRAINT_TOLERANCE * abs(target):
            error += ((val - target) / normalizer) ** 2
        elif rule == ">=" and val < target - config.CONSTRAINT_TOLERANCE * abs(target):
            error += ((val - target) / normalizer) ** 2
    
    # Enhanced cost normalization
    cost = np.dot(weights, cost_vector)
    if config.COST_NORMALIZATION_METHOD == 'minmax':
        cost_min, cost_max = np.min(cost_vector), np.max(cost_vector)
        normalized_cost = (cost - cost_min) / (cost_max - cost_min) if cost_max > cost_min else 0
    else:  # zscore normalization
        cost_mean, cost_std = np.mean(cost_vector), np.std(cost_vector)
        normalized_cost = (cost - cost_mean) / cost_std if cost_std > 0 else 0
    
    return error + lam * normalized_cost

def robust_pre_filter_improved(compound_data, target_dict, property_weights, top_n=150, 
                              cost_weight=0.3, config=None):
    """Enhanced pre-filtering with consistent scoring logic"""
    
    if config is None:
        config = OptimizationConfig()
    
    try:
        scores = []
        prop_matrix = compound_data[properties].values
        cost_vector = compound_data['cost'].values
        reo_values = compound_data[REO_PROPERTY].values if REO_PROPERTY in compound_data.columns else np.zeros(len(compound_data))
        
        # Use same normalization as main optimization
        if config.PROPERTY_NORMALIZATION_METHOD == 'std':
            property_stds = np.std(prop_matrix, axis=0)
            property_stds[property_stds == 0] = 1.0
        
        # Cost normalization consistent with main optimization
        if config.COST_NORMALIZATION_METHOD == 'minmax':
            cost_min, cost_max = np.min(cost_vector), np.max(cost_vector)
            cost_range = cost_max - cost_min if cost_max > cost_min else 1.0
        
        for idx, row in enumerate(prop_matrix):
            # Check restriction violation first
            if reo_values[idx] > 0:
                # Apply penalty for restricted compounds
                restriction_penalty = config.RESTRICTION_VIOLATION_PENALTY / 1000  # Scaled for pre-filtering
            else:
                restriction_penalty = 0
            
            # Use same constraint logic as blend_error
            constraint_score = 0
            total_weight = 0
            
            for i, prop in enumerate(properties):
                if prop in target_dict and prop in property_weights:
                    val = row[i]
                    target = target_dict[prop]["value"]
                    rule = target_dict[prop]["type"]
                    weight = property_weights[prop]
                    
                    # Same constraint evaluation as blend_error
                    if config.PROPERTY_NORMALIZATION_METHOD == 'std':
                        normalizer = property_stds[i]
                    else:
                        normalizer = abs(target) if target != 0 else 1.0
                    
                    if rule == "=":
                        deviation = abs(val - target)
                        if deviation > config.CONSTRAINT_TOLERANCE * abs(target):
                            constraint_score += weight * (deviation / normalizer) ** 2
                    elif rule == "<=" and val > target + config.CONSTRAINT_TOLERANCE * abs(target):
                        constraint_score += weight * ((val - target) / normalizer) ** 2
                    elif rule == ">=" and val < target - config.CONSTRAINT_TOLERANCE * abs(target):
                        constraint_score += weight * ((val - target) / normalizer) ** 2
                    
                    total_weight += weight
            
            # Average constraint score
            avg_constraint_score = constraint_score / max(total_weight, 1)
            
            # Cost score with consistent normalization
            if config.COST_NORMALIZATION_METHOD == 'minmax':
                cost_score = (cost_vector[idx] - cost_min) / cost_range
            else:
                cost_mean, cost_std = np.mean(cost_vector), np.std(cost_vector)
                cost_score = (cost_vector[idx] - cost_mean) / cost_std if cost_std > 0 else 0
            
            # Combined score (lower is better)
            combined_score = (1 - cost_weight) * avg_constraint_score + cost_weight * cost_score + restriction_penalty
            
            scores.append((idx, combined_score))
        
        # Sort and return top candidates
        scores.sort(key=lambda x: x[1])
        top_indices = [idx for idx, _ in scores[:top_n]]
        
        return compound_data.iloc[top_indices].reset_index(drop=True)
    
    except Exception as e:
        st.error(f"Pre-filtering failed: {str(e)}")
        return compound_data
    
 

def stable_multi_start_optimization(prop_matrix, constraint_dict, cost_vector, lam, 
                                   reo_values, config=None,n_starts=5):
    """Enhanced optimization with configurable parameters"""
    
    if config is None:
        config = OptimizationConfig()
    
    best_results = []
    n = len(prop_matrix)
    if n == 0:
        return []
    
    bounds = [(0, 1)] * n
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    
    # Calculate property standards
    if config.PROPERTY_NORMALIZATION_METHOD == 'std':
        property_stds = np.std(prop_matrix, axis=0)
        property_stds[property_stds == 0] = 1.0
    else:
        property_stds = None
    
    # Adaptive number of starts based on problem complexity
    n_starts = max(config.DEFAULT_N_STARTS, min(10, n // 20))
    
    for start in range(n_starts):
        # Diverse initialization strategies
        if start == 0:
            initial = np.ones(n) / n  # Uniform start
        elif start == 1:
            # Cost-weighted initialization
            cost_weights = 1.0 / (cost_vector + 1e-6)
            initial = cost_weights / np.sum(cost_weights)
        else:
            # Random initialization
            initial = np.random.dirichlet(np.ones(n))
        
        try:
            res = minimize(
                blend_error,
                initial,
                args=(prop_matrix, constraint_dict, cost_vector, lam, property_stds, reo_values, True, config),
                method='SLSQP',
                bounds=bounds,
                constraints=[constraints],
                options={
                    'maxiter': config.MAX_ITERATIONS,
                    'ftol': config.CONVERGENCE_TOLERANCE,
                    'disp': False
                }
            )
            
            if res.success and not np.isnan(res.fun):
                best_results.append(res)
                
        except Exception as e:
            continue
    
    # Sort by objective value
    best_results.sort(key=lambda x: x.fun)
    return best_results

def optimize_lots_for_blend(lot_data, target_proportions, target_dict, lambda_val=1.0, config=None):
    """Enhanced lot optimization with configurable constraint tolerances"""
    
    if config is None:
        config = OptimizationConfig()
    
    try:
        # Get compounds used in blend
        used_compounds = list(target_proportions.keys())
        
        # Filter lots for used compounds
        available_lots = lot_data[lot_data['Compound Name'].isin(used_compounds)].copy()
        
        if available_lots.empty:
            st.warning("No lots available for the selected compounds")
            return None, None, None
        
        # Prepare lot data
        lot_prop_matrix = available_lots[properties].values
        lot_cost_vector = available_lots['cost'].values
        lot_compound_names = available_lots['Compound Name'].values
        lot_reo_values = available_lots[REO_PROPERTY].values if REO_PROPERTY in available_lots.columns else np.zeros(len(available_lots))
        
        n_lots = len(available_lots)
        
        # Create compound mapping
        compound_to_indices = {}
        for i, compound in enumerate(lot_compound_names):
            if compound not in compound_to_indices:
                compound_to_indices[compound] = []
            compound_to_indices[compound].append(i)
        
        # Set up optimization problem
        bounds = [(0, 1)] * n_lots
        
        # Constraint: sum of weights = 1
        def weight_constraint(w):
            return np.sum(w) - 1
        
        # Enhanced constraint handling with configurable tolerance
        def compound_proportion_constraints(w):
            constraints = []
            for compound, target_prop in target_proportions.items():
                if compound in compound_to_indices:
                    indices = compound_to_indices[compound]
                    actual_prop = np.sum([w[i] for i in indices])
                    
                    # Use configurable tolerance
                    tolerance = config.PROPORTION_TOLERANCE * target_prop
                    constraints.append(actual_prop - target_prop + tolerance)  # Upper bound
                    constraints.append(target_prop - actual_prop + tolerance)  # Lower bound
            return constraints
        
        # Combine constraints
        constraints = [
            {"type": "eq", "fun": weight_constraint},
            {"type": "ineq", "fun": lambda w: compound_proportion_constraints(w)}
        ]
        
        # Calculate property standards for lot data
        lot_property_stds = np.std(lot_prop_matrix, axis=0)
        lot_property_stds[lot_property_stds == 0] = 1.0
        
        # Multi-start optimization
        best_result = None
        best_score = float('inf')
        
        for start in range(max(3, min(8, len(target_proportions)))):  # Adaptive starts
            # Initialize based on target proportions
            initial = np.zeros(n_lots)
            for compound, target_prop in target_proportions.items():
                if compound in compound_to_indices:
                    indices = compound_to_indices[compound]
                    if indices:
                        prop_per_lot = target_prop / len(indices)
                        for idx in indices:
                            initial[idx] = prop_per_lot
                            
            # Add some randomness for non-first starts
            if start > 0:
                noise = np.random.normal(0, 0.01, n_lots)
                initial = np.clip(initial + noise, 0, 1)
                initial = initial / np.sum(initial)  # Normalize
            
            try:
                res = minimize(
                    blend_error,
                    initial,
                    args=(lot_prop_matrix, target_dict, lot_cost_vector, lambda_val, 
                          lot_property_stds, lot_reo_values, True, config),
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={
                        'maxiter': config.MAX_ITERATIONS,
                        'ftol': config.CONVERGENCE_TOLERANCE,
                        'disp': False
                    }
                )
                
                if res.success and not np.isnan(res.fun) and res.fun < best_score:
                    best_result = res
                    best_score = res.fun
                    
            except Exception as e:
                continue
        
        if best_result is None:
            st.warning("Lot optimization failed, using proportional distribution")
            # Fallback to proportional distribution
            lot_weights = np.zeros(n_lots)
            for compound, target_prop in target_proportions.items():
                if compound in compound_to_indices:
                    indices = compound_to_indices[compound]
                    if indices:
                        prop_per_lot = target_prop / len(indices)
                        for idx in indices:
                            lot_weights[idx] = prop_per_lot
            
            final_cost = np.sum(lot_weights * lot_cost_vector)
            final_props = np.dot(lot_weights, lot_prop_matrix)
            
            return lot_weights, final_cost, final_props
        
        # Return optimized results
        final_cost = np.sum(best_result.x * lot_cost_vector)
        final_props = np.dot(best_result.x, lot_prop_matrix)
        
        return best_result.x, final_cost, final_props
    except Exception as e:
        st.error(f"Lot optimization failed: {str(e)}")
        return None, None, None

def calculate_blend_properties(weights, prop_matrix):
    """Calculate resulting blend properties with validation"""
    if len(weights) == 0 or len(prop_matrix) == 0:
        return np.zeros(len(properties))
    return np.dot(weights, prop_matrix)

def check_restricted_compounds(weights, reo_values, compound_names, config=None):
    """Enhanced restriction checking with configurable thresholds"""
    
    if config is None:
        config = OptimizationConfig()
    
    if len(weights) == 0:
        return 0.0, []
    
    restricted_weight = 0
    restricted_compounds = []
    
    for i, weight in enumerate(weights):
        if weight > config.WEIGHT_SIGNIFICANCE_THRESHOLD and reo_values[i] > 0:
            restricted_weight += weight
            restricted_compounds.append(compound_names[i])
    
    return restricted_weight * 100, restricted_compounds

# Phase 1: Ideal Blend Optimization
if st.session_state.optimization_stage == 'initial':
    if st.button("🚀 Generate Top Blends", type="primary"):
        with st.spinner("Optimizing blends with improved pre-filtering..."):
            # Apply filters
            working_data = compound_data.copy()
            
            # Exclude compounds
            if exclude_compounds:
                working_data = working_data[~working_data['Compound Name'].isin(exclude_compounds)]
            
            # Improved pre-filtering with consistent scoring
            if len(working_data) > top_n:
                working_data = robust_pre_filter_improved(
                    working_data, 
                    target_dict, 
                    property_weights, 
                    top_n, 
                    cost_weight_prefilter
                )
                st.info(f"Pre-filtered to {len(working_data)} compounds using improved scoring")
            
            # Prepare matrices
            prop_matrix = working_data[properties].values
            cost_vector = working_data['cost'].values
            reo_values = working_data[REO_PROPERTY].values
            compound_names = working_data['Compound Name'].values
            
            n = len(working_data)
            if n == 0:
                st.error("No compounds available after filtering!")
                st.stop()
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            start_time = time.time()
            
            # Run optimization
            status_text.text("Running robust optimization...")
            progress_bar.progress(0.3)
            results = stable_multi_start_optimization(prop_matrix, target_dict, cost_vector, lambda_val, reo_values, n_starts=5)
            
            progress_bar.progress(1.0)
            end_time = time.time()
            
            if results:
                st.success(f"Optimization completed in {end_time - start_time:.2f} seconds")
                
                # Process top blends
                top_blends = []
                for i, res in enumerate(results[:3]):  # Get top 3 blends
                    # Skip solutions with no significant compounds
                    if np.max(res.x) < 0.01:
                        continue
                    
                    # Filter insignificant compounds
                    significant_weights = np.where(res.x > 0.01, res.x, 0)
                    total_significant = np.sum(significant_weights)
                    
                    # Normalize weights if needed
                    if total_significant > 0:
                        normalized_weights = significant_weights / total_significant
                    else:
                        normalized_weights = res.x
                    
                    # Calculate properties
                    blend_props = calculate_blend_properties(normalized_weights, prop_matrix)
                    
                    # Check restrictions
                    restricted_pct, restricted_comps = check_restricted_compounds(normalized_weights, reo_values, compound_names)
                    
                    # Get selected compounds
                    working_data_copy = working_data.copy()
                    working_data_copy['Weight'] = normalized_weights
                    selected_compounds = working_data_copy[working_data_copy['Weight'] > 0.01]
                    
                    # Calculate total cost
                    total_cost = np.sum(normalized_weights * cost_vector)
                    
                    # Calculate deviations
                    deviations = {}
                    for j, prop in enumerate(properties):
                        target_val = target_dict[prop]['value']
                        blend_val = blend_props[j]
                        if target_val != 0:
                            deviation_pct = ((blend_val - target_val) / target_val) * 100
                        else:
                            deviation_pct = 0.0
                        deviations[prop] = deviation_pct
                    
                    blend_info = {
                        'id': i + 1,
                        'result': res,
                        'properties': blend_props,
                        'cost': total_cost,
                        'compounds': selected_compounds,
                        'restricted_pct': restricted_pct,
                        'restricted_compounds': restricted_comps,
                        'deviations': deviations,
                        'working_data': working_data_copy
                    }
                    
                    top_blends.append(blend_info)
                
                if not top_blends:
                    st.error("All blends were invalid. Try relaxing constraints.")
                else:
                    # Store in session state
                    st.session_state.top_blends = top_blends
                    st.session_state.optimization_stage = 'blend_selection'
                    st.rerun()
            
            else:
                st.error("Optimization failed! Try adjusting parameters.")
            
            status_text.empty()
            progress_bar.empty()

# Phase 2: Blend Selection
elif st.session_state.optimization_stage == 'blend_selection':
    st.subheader("🏆 Top Optimized Blends")
    
    if not st.session_state.top_blends:
        st.error("No valid blends found. Try generating again.")
        if st.button("🔄 Generate New Blends", type="primary"):
            st.session_state.optimization_stage = 'initial'
            st.rerun()
        st.stop()
    
    # Display top blends
    for i, blend in enumerate(st.session_state.top_blends):
        
        with st.expander(f"🧪 Blend {blend['id']} - Cost: ${blend['cost']:.2f}", expanded=i==0):
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                st.metric("Total Cost", f"${blend['cost']:.2f}")
                if blend['restricted_pct'] > RESTRICTED_LIMIT:
                    st.warning(f"Restricted Compounds: {blend['restricted_pct']:.1f}%")
                else:
                    st.metric("Restricted Compounds", f"{blend['restricted_pct']:.1f}%")
                st.metric("Compounds Used", len(blend['compounds']))
            
            with col2:
                st.write("**Selected Compounds:**")
                if blend['compounds'].empty:
                    st.warning("No compounds selected")
                else:
                    display_compounds = blend['compounds'][['Compound Name', 'Weight', 'cost']].copy()
                    display_compounds['Weight %'] = (display_compounds['Weight'] * 100).round(1)
                    display_compounds = display_compounds.sort_values('Weight %', ascending=False)
                    st.dataframe(display_compounds[['Compound Name', 'Weight %', 'cost']], 
                                 use_container_width=True, height=300)
            
            with col3:
                st.write("**Property Deviations:**")
                dev_df = pd.DataFrame.from_dict(blend['deviations'], orient='index', columns=['Deviation %'])
                dev_df['Status'] = dev_df['Deviation %'].apply(lambda x: '✅' if abs(x) < 5 else '⚠️')
                st.dataframe(dev_df, use_container_width=True, height=300)
            
            # Visualization
            if not blend['compounds'].empty:
                fig = go.Figure()
                
                target_vals = [target_dict[p]['value'] for p in properties]
                blend_vals = blend['properties']
                
                fig.add_trace(go.Scatter(
                    x=properties,
                    y=target_vals,
                    mode='lines+markers',
                    name='Target',
                    line=dict(color='blue', width=3)
                ))
                
                fig.add_trace(go.Scatter(
                    x=properties,
                    y=blend_vals,
                    mode='lines+markers',
                    name='Blend',
                    line=dict(color='red', width=2)
                ))
                
                fig.update_layout(
                    title=f"Blend {blend['id']} - Property Comparison",
                    xaxis_title="Property",
                    yaxis_title="Value",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Selection button
            if st.button(f"Select Blend {blend['id']} for Lot Optimization", key=f"select_{i}"):
                st.session_state.selected_blend = blend
                st.session_state.optimization_stage = 'lot_optimization'
                st.rerun()
    
    # Reset button
    if st.button("🔄 Generate New Blends", type="secondary"):
        st.session_state.optimization_stage = 'initial'
        st.session_state.top_blends = []
        st.session_state.selected_blend = None
        st.rerun()

# Phase 3: Lot Optimization
elif st.session_state.optimization_stage == 'lot_optimization':
    st.subheader("🏭 Lot-Level Optimization")
    
    selected_blend = st.session_state.selected_blend
    
    st.info(f"Optimizing lots for Blend {selected_blend['id']} (Cost: ${selected_blend['cost']:.2f})")
    
    # Show selected blend summary
    with st.expander("📋 Selected Blend Summary", expanded=True):
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.write("**Compound Proportions:**")
            if selected_blend['compounds'].empty:
                st.warning("No compounds in selected blend")
            else:
                compounds_df = selected_blend['compounds'][['Compound Name', 'Weight']].copy()
                compounds_df['Weight %'] = (compounds_df['Weight'] * 100).round(1)
                st.dataframe(compounds_df[['Compound Name', 'Weight %']], use_container_width=True)
        
        with col2:
            st.write("**Target Properties:**")
            target_summary = pd.DataFrame.from_dict(target_dict, orient='index')
            st.dataframe(target_summary, use_container_width=True)
    
    # Show available lots before optimization
    if lot_data is not None and not selected_blend['compounds'].empty:
        target_proportions = selected_blend['compounds'].set_index('Compound Name')['Weight'].to_dict()
        available_lots = lot_data[lot_data['Compound Name'].isin(target_proportions.keys())].copy()
        
        with st.expander("📦 Available Lots", expanded=False):
            st.write(f"**Available lots for selected compounds ({len(available_lots)} lots):**")
            display_lots = available_lots[['Compound Name', 'Lot'] + properties + ['cost']].copy()
            st.dataframe(display_lots, use_container_width=True)
    
    # Optimization button
    if st.button("🔍 Optimize Lots", type="primary"):
        with st.spinner("Running actual lot-level optimization..."):
            
            # Validate we have lot data
            if lot_data is None:
                st.error("Lot data not available")
                st.stop()
                
            # Validate required columns in lot data
            if 'Compound Name' not in lot_data.columns:
                st.error("Lot data missing 'Compound Name' column")
                st.stop()
                
            # Ensure cost column exists
            if 'cost' not in lot_data.columns:
                np.random.seed(42)
                lot_data['cost'] = np.random.randint(100, 1000, len(lot_data))
                st.warning("Generated cost column for lot data")
            
            # Get compounds used in selected blend
            if selected_blend['compounds'].empty:
                st.error("No compounds in selected blend")
                st.stop()
                
            # Extract target proportions
            target_proportions = selected_blend['compounds'].set_index('Compound Name')['Weight'].to_dict()
            
            # Run improved lot optimization
            lot_weights, final_cost, final_blend_props = optimize_lots_for_blend(
                lot_data, 
                target_proportions, 
                target_dict, 
                lambda_val
            )
            
            if lot_weights is None:
                st.error("Lot optimization failed")
                st.stop()
            
            # Filter lots for display
            available_lots = lot_data[lot_data['Compound Name'].isin(target_proportions.keys())].copy()
            
            # SAVE RESULTS TO SESSION STATE
            st.session_state.lot_results = {
                'weights': lot_weights,
                'cost': final_cost,
                'properties': final_blend_props,
                'available_lots': available_lots,
                'target_proportions': target_proportions,
                'ideal_cost': selected_blend['cost'],
                'ideal_properties': selected_blend['properties']
            }
            
            st.success("Lot optimization completed successfully!")
            st.rerun()  # Rerun to show results
    
    # DISPLAY RESULTS (Outside button handler)
    if 'lot_results' in st.session_state:
        lot_results = st.session_state.lot_results
        
        st.subheader("📊 Lot Optimization Results")
        
        # Cost comparison
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            st.metric("Ideal Blend Cost", f"${lot_results['ideal_cost']:.2f}")
        with col2:
            st.metric("Actual Lot Cost", f"${lot_results['cost']:.2f}")
        with col3:
            cost_increase = ((lot_results['cost'] - lot_results['ideal_cost']) / lot_results['ideal_cost']) * 100
            st.metric("Cost Increase", f"{cost_increase:.1f}%")
        
        # Selected lots table
        st.subheader("🎯 Selected Lots")
        available_lots = lot_results['available_lots'].copy()
        available_lots['Selected_Weight'] = lot_results['weights']
        available_lots['Weight_%'] = (available_lots['Selected_Weight'] * 100).round(2)
        
        # Filter to show only selected lots
        selected_lots = available_lots[available_lots['Selected_Weight'] > 0.001].copy()
        selected_lots = selected_lots.sort_values('Weight_%', ascending=False)
        
        if not selected_lots.empty:
            display_cols = ['Compound Name', 'Lot', 'Weight_%', 'cost'] + properties
            st.dataframe(selected_lots[display_cols], use_container_width=True)
            
         
            
            
            # Property comparison
            st.subheader("🔬 Property Comparison")
            comparison_df = pd.DataFrame({
                'Property': properties,
                'Target': [target_dict[prop]['value'] for prop in properties],
                'Ideal_Blend': lot_results['ideal_properties'],
                'Actual_Lots': lot_results['properties']
            })
            
            # Calculate deviations
            comparison_df['Ideal_Deviation_%'] = ((comparison_df['Ideal_Blend'] - comparison_df['Target']) / comparison_df['Target'] * 100).round(2)
            comparison_df['Lot_Deviation_%'] = ((comparison_df['Actual_Lots'] - comparison_df['Target']) / comparison_df['Target'] * 100).round(2)
            
            st.dataframe(comparison_df, use_container_width=True)
            
            # Property comparison chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=properties,
                y=comparison_df['Target'],
                mode='lines+markers',
                name='Target',
                line=dict(color='blue', width=3)
            ))
            fig.add_trace(go.Scatter(
                x=properties,
                y=comparison_df['Ideal_Blend'],
                mode='lines+markers',
                name='Ideal Blend',
                line=dict(color='green', width=2)
            ))
            fig.add_trace(go.Scatter(
                x=properties,
                y=comparison_df['Actual_Lots'],
                mode='lines+markers',
                name='Actual Lots',
                line=dict(color='red', width=2)
            ))
            
            fig.update_layout(
                title="Property Comparison: Target vs Ideal vs Actual",
                xaxis_title="Property",
                yaxis_title="Value",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.warning("No lots were selected in the optimization")
    
    # Navigation buttons
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("← Back to Blend Selection", type="secondary"):
            st.session_state.optimization_stage = 'blend_selection'
            if 'lot_results' in st.session_state:
                del st.session_state.lot_results
            st.rerun()

    with col2:
        if st.button("🔄 Start Over", type="secondary"):
            st.session_state.optimization_stage = 'initial'
            st.session_state.top_blends = []
            st.session_state.selected_blend = None
            if 'lot_results' in st.session_state:
                del st.session_state.lot_results
            st.rerun()