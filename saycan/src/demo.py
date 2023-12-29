# Define and reset environment.
config = {'pick':  ['yellow block', 'green block', 'blue block'],
          'place': ['yellow bowl', 'green bowl', 'blue bowl']}

np.random.seed(42)
obs = env.reset(config)
img = env.get_camera_image()
plt.imshow(img)
plt.show()

user_input = 'Pick the yellow block and place it on the blue bowl.'  #@param {type:"string"}

# Show camera image before pick and place.

def run_cliport(obs, text):
  before = env.get_camera_image()
  prev_obs = obs['image'].copy()

  # Tokenize text and get CLIP features.
  text_tokens = clip.tokenize(text).cuda()
  with torch.no_grad():
    text_feats = clip_model.encode_text(text_tokens).float()
  text_feats /= text_feats.norm(dim=-1, keepdim=True)
  text_feats = np.float32(text_feats.cpu())

  # Normalize image and add batch dimension.
  img = obs['image'][None, ...] / 255
  img = np.concatenate((img, coords[None, ...]), axis=3)

  # Run Transporter Nets to get pick and place heatmaps.
  batch = {'img': jnp.float32(img), 'text': jnp.float32(text_feats)}
  pick_map, place_map = eval_step(optim.target, batch)
  pick_map, place_map = np.float32(pick_map), np.float32(place_map)

  # Get pick position.
  pick_max = np.argmax(np.float32(pick_map)).squeeze()
  pick_yx = (pick_max // 224, pick_max % 224)
  pick_yx = np.clip(pick_yx, 20, 204)
  pick_xyz = obs['xyzmap'][pick_yx[0], pick_yx[1]]

  # Get place position.
  place_max = np.argmax(np.float32(place_map)).squeeze()
  place_yx = (place_max // 224, place_max % 224)
  place_yx = np.clip(place_yx, 20, 204)
  place_xyz = obs['xyzmap'][place_yx[0], place_yx[1]]

  # Step environment.
  act = {'pick': pick_xyz, 'place': place_xyz}
  obs, _, _, _ = env.step(act)

  # Show pick and place action.
  plt.title(text)
  plt.imshow(prev_obs)
  plt.arrow(pick_yx[1], pick_yx[0], place_yx[1]-pick_yx[1], place_yx[0]-pick_yx[0], color='w', head_starts_at_zero=False, head_width=7, length_includes_head=True)
  plt.show()

  # Show debug plots.
  plt.subplot(1, 2, 1)
  plt.title('Pick Heatmap')
  plt.imshow(pick_map.reshape(224, 224))
  plt.subplot(1, 2, 2)
  plt.title('Place Heatmap')
  plt.imshow(place_map.reshape(224, 224))
  plt.show()

  # Show video of environment rollout.
  debug_clip = ImageSequenceClip(env.cache_video, fps=25)
  display(debug_clip.ipython_display(autoplay=1, loop=1, center=False))
  env.cache_video = []

  # Show camera image after pick and place.
  plt.subplot(1, 2, 1)
  plt.title('Before')
  plt.imshow(before)
  plt.subplot(1, 2, 2)
  plt.title('After')
  after = env.get_camera_image()
  plt.imshow(after)
  plt.show()

  # return pick_xyz, place_xyz, pick_map, place_map, pick_yx, place_yx
  return obs


# pick_xyz, place_xyz, pick_map, place_map, pick_yx, place_yx = 
obs = run_cliport(obs, user_input)
