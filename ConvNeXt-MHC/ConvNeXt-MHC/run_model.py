from utils import input_matrix_generation, get_predict
from model import *
from utils import *


def run_model():
    # 数据保存在./data_set中/，为了节约时间，已经将多肽转换为了9mer数据
    # 如果需要添加新的，参照example文件进行添加，并修改下述文件名
    # os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
    
    print("=" * 70)
    print("ConvNeXt-MHC Demo: Using example.csv and test datasets")
    print("=" * 70)
    
    # BA score predict - using available test set
    print("\n[1/3] Testing BA (Binding Affinity) Prediction...")
    data_file_name = "AF_Test_set.csv"  # Changed to available file
    print(f"Loading data from: {data_file_name}")
    input_matrix_BA = input_matrix_generation(data_file_name)
    print(f"Generated input matrix shape: {input_matrix_BA.shape}")
    
    print("Loading ConvNeXt-MHC_BA model...")
    ConvNeXt_MHC_BA = convnext_10_10(2)
    ConvNeXt_MHC_BA.build((1, 20, 9, 21))
    ConvNeXt_MHC_BA.load_weights("./save_model_weight/ConvNeXt-MHC_BA.h5")
    
    print("Running predictions...")
    BA_ans = ConvNeXt_MHC_BA.predict(input_matrix_BA)[:, 1]
    print(f"\nBA Predictions (first 10): {BA_ans[:10]}")
    print(f"BA Predictions Stats: min={BA_ans.min():.3f}, max={BA_ans.max():.3f}, mean={BA_ans.mean():.3f}, std={BA_ans.std():.3f}")
    
    # Save BA results
    np.save("./ba_predictions.npy", BA_ans)
    print(f"✓ Saved {len(BA_ans)} BA predictions to ba_predictions.npy")

    # AP score predict - using available test set
    print("\n[2/3] Testing AP (Antigen Presentation) Prediction...")
    data_file_name = "MS_Test_set.csv"  # Changed to available file
    print(f"Loading data from: {data_file_name}")
    input_matrix_AP = input_matrix_generation(data_file_name)
    print(f"Generated input matrix shape: {input_matrix_AP.shape}")
    
    print("Loading ConvNeXt-MHC_AP model...")
    ConvNeXt_MHC_AP = convnext_10_10(2)
    ConvNeXt_MHC_AP.build((1, 20, 9, 21))
    ConvNeXt_MHC_AP.load_weights("./save_model_weight/ConvNeXt-MHC_AP.h5")  # Fixed: use AP weights
    
    print("Running predictions...")
    AP_ans = get_predict(ConvNeXt_MHC_AP.predict(input_matrix_AP))
    print(f"\nAP Predictions (first 10): {AP_ans[:10]}")
    print(f"AP Predictions: {np.sum(AP_ans == 1)} presented / {len(AP_ans)} total ({100*np.sum(AP_ans == 1)/len(AP_ans):.1f}%)")
    
    # Save AP results
    np.save("./ap_predictions.npy", AP_ans)
    with open("./ap_predictions.txt", "w") as f:
        for AP in AP_ans:
            f.write("{}\n".format(AP))
    print(f"✓ Saved {len(AP_ans)} AP predictions to ap_predictions.txt and .npy")
    
    # Example prediction demo
    print("\n[3/3] Demo on example.csv (small dataset)...")
    data_file_name = "example.csv"
    print(f"Loading data from: {data_file_name}")
    input_matrix_example = input_matrix_generation(data_file_name)
    print(f"Generated input matrix shape: {input_matrix_example.shape}")
    
    print("Running BA predictions on example data...")
    example_BA = ConvNeXt_MHC_BA.predict(input_matrix_example)[:, 1]
    print(f"Example BA predictions: {example_BA[:10]}")
    
    print("\n" + "=" * 70)
    print("✓ All predictions complete!")
    print("=" * 70)
    print(f"\nOutput files:")
    print(f"  - ba_predictions.npy ({len(BA_ans)} predictions)")
    print(f"  - ap_predictions.npy ({len(AP_ans)} predictions)")
    print(f"  - ap_predictions.txt ({len(AP_ans)} predictions)")
    print(f"\nTo use on your own data:")
    print(f"  1. Create CSV with columns: allele, peptide, 9mer")
    print(f"  2. Change data_file_name in run_model.py")
    print(f"  3. Run: python run_model.py")



run_model()
