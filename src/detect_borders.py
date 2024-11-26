def gaussian_blur(img: np.array, sigma: float, filter_shape: None = None, verbose: bool = False) -> np.array:
    # TODO If not given, compute the filter shape 
    if filter_shape == None:
        filter_l = int(2 * sigma) + 1
    else:
        filter_l = filter_shape[0]
    
    # TODO Create the filter coordinates matrices
    y, x = np.mgrid[-filter_l//2 + 1:filter_l//2 + 1, -filter_l//2 + 1:filter_l//2 + 1]
    
    # TODO Define the formula that goberns the filter
    formula = np.exp(-((x**2 + y**2) / (2.0 * sigma**2))) / (2 * np.pi * sigma**2)
    gaussian_filter = formula / formula.sum()
    
    # TODO Process the image
    gb_img = cv2.filter2D(img, -1, gaussian_filter)
    
    if verbose:
        show_image(img=gb_img, img_name=f"Gaussian Blur: Sigma = {sigma}")
    
    return gaussian_filter, gb_img.astype(np.uint8)

def sobel_edge_detector(img: np.array, filter: np.array, gauss_sigma: float, gauss_filter_shape: None = None, verbose: bool = False) -> np.array:
    # TODO Transform the img to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # TODO Get a blurry img to improve edge detections
    blurred = gaussian_blur(img=gray_img, sigma=gauss_sigma, filter_shape=gauss_filter_shape, verbose=verbose)
    blurred = blurred[1]
    
    # Re-scale
    blurred = blurred/255
    
    # TODO Get vertical edges
    v_edges = cv2.filter2D(blurred, -1, filter)
    
    # TODO Transform the filter to get the orthogonal edges
    filter = np.flip(filter.T, axis=0)
    
    # TODO Get horizontal edges
    h_edges = cv2.filter2D(blurred, -1, filter)
    
    # TODO Get edges
    sobel_edges_img = np.hypot(v_edges, h_edges)
    
    # Get edges angle
    theta = np.arctan2(h_edges, v_edges)
    
    # Visualize if needed
    if verbose:
        show_image(img=sobel_edges_img, img_name="Sobel Edges")
    
    return np.squeeze(sobel_edges_img), np.squeeze(theta)

def canny_edge_detector(img: np.array, sobel_filter: np.array, gauss_sigma: float, gauss_filter_shape: List = None, verbose: bool = False):
    # TODO Call the method sobel_edge_detector()
    sobel_edges_imgs, theta = sobel_edge_detector(img, sobel_filter, gauss_sigma, gauss_filter_shape, verbose)
    
    # TODO Use NMS to refine edges
    canny_edges_img = non_max_suppression(sobel_edges_imgs, theta)
    
    if verbose:
        show_image(canny_edges_img, img_name="Canny Edges")
        
    return canny_edges_img

gauss_sigma = 3
gb_imgs = [gaussian_blur(img, gauss_sigma, verbose=False) for img in imgs]
show_image(gb_imgs[0][1])

# TODO Define a sigma value
gauss_sigma = 3

# TODO Define the Sobel filter
sobel_filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

# TODO Get the edges detected by Sobel using a list comprehension
sobel_edges_imgs = [sobel_edge_detector(img, sobel_filter, gauss_sigma, verbose=False) for img in imgs]
for i in range(len(sobel_edges_imgs)):
    show_image(sobel_edges_imgs[i][0], f"Edges: {i}")

# TODO Define Sobel filter
sobel_filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

# TODO Define a sigma value for Gauss
gauss_sigma = 1

# TODO Define a Gauss filter shape
gauss_filter_shape = [3, 3]

# TODO Get the edges detected by Canny using a list comprehension
canny_edges_imgs = [canny_edge_detector(img, sobel_filter, gauss_sigma, gauss_filter_shape, verbose=False) for img in imgs]
for i in range(len(canny_edges_imgs)):
    show_image(canny_edges_imgs[i], f"Canny Edges: {i}")