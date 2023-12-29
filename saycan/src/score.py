#@title LLM Scoring

def gpt3_call(engine="text-ada-001", prompt="", max_tokens=128, temperature=0, 
              logprobs=1, echo=False):
  full_query = ""
  for p in prompt:
    full_query += p
  id = tuple((engine, full_query, max_tokens, temperature, logprobs, echo))
  if id in LLM_CACHE.keys():
    print('cache hit, returning')
    response = LLM_CACHE[id]
  else:
    response = openai.Completion.create(engine=engine, 
                                        prompt=prompt, 
                                        max_tokens=max_tokens, 
                                        temperature=temperature,
                                        logprobs=logprobs,
                                        echo=echo)
    LLM_CACHE[id] = response
  return response

def gpt3_scoring(query, options, engine="text-ada-001", limit_num_options=None, option_start="\n", verbose=False, print_tokens=False):
  if limit_num_options:
    options = options[:limit_num_options]
  verbose and print("Scoring", len(options), "options")
  gpt3_prompt_options = [query + option for option in options]
  response = gpt3_call(
      engine=engine, 
      prompt=gpt3_prompt_options, 
      max_tokens=0,
      logprobs=1, 
      temperature=0,
      echo=True,)
  
  scores = {}
  for option, choice in zip(options, response["choices"]):
    tokens = choice["logprobs"]["tokens"]
    token_logprobs = choice["logprobs"]["token_logprobs"]

    total_logprob = 0
    for token, token_logprob in zip(reversed(tokens), reversed(token_logprobs)):
      print_tokens and print(token, token_logprob)
      if option_start is None and not token in option:
        break
      if token == option_start:
        break
      total_logprob += token_logprob
    scores[option] = total_logprob

  for i, option in enumerate(sorted(scores.items(), key=lambda x : -x[1])):
    verbose and print(option[1], "\t", option[0])
    if i >= 10:
      break

  return scores, response

def make_options(pick_targets=None, place_targets=None, options_in_api_form=True, termination_string="done()"):
  if not pick_targets:
    pick_targets = PICK_TARGETS
  if not place_targets:
    place_targets = PLACE_TARGETS
  options = []
  for pick in pick_targets:
    for place in place_targets:
      if options_in_api_form:
        option = "robot.pick_and_place({}, {})".format(pick, place)
      else:
        option = "Pick the {} and place it on the {}.".format(pick, place)
      options.append(option)

  options.append(termination_string)
  print("Considering", len(options), "options")
  return options

query = "To pick the blue block and put it on the red block, I should:\n"
options = make_options(PICK_TARGETS, PLACE_TARGETS)
scores, response = gpt3_scoring(query, options, engine=ENGINE, limit_num_options=5, option_start='\n', verbose=True)

#@title Helper Functions

def build_scene_description(found_objects, block_name="box", bowl_name="circle"):
  scene_description = f"objects = {found_objects}"
  scene_description = scene_description.replace(block_name, "block")
  scene_description = scene_description.replace(bowl_name, "bowl")
  scene_description = scene_description.replace("'", "")
  return scene_description

def step_to_nlp(step):
  step = step.replace("robot.pick_and_place(", "")
  step = step.replace(")", "")
  pick, place = step.split(", ")
  return "Pick the " + pick + " and place it on the " + place + "."

def normalize_scores(scores):
  max_score = max(scores.values())  
  normed_scores = {key: np.clip(scores[key] / max_score, 0, 1) for key in scores}
  return normed_scores

def plot_saycan(llm_scores, vfs, combined_scores, task, correct=True, show_top=None):
  if show_top:
    top_options = nlargest(show_top, combined_scores, key = combined_scores.get)
    # add a few top llm options in if not already shown
    top_llm_options = nlargest(show_top // 2, llm_scores, key = llm_scores.get)
    for llm_option in top_llm_options:
      if not llm_option in top_options:
        top_options.append(llm_option)
    llm_scores = {option: llm_scores[option] for option in top_options}
    vfs = {option: vfs[option] for option in top_options}
    combined_scores = {option: combined_scores[option] for option in top_options}

  sorted_keys = dict(sorted(combined_scores.items()))
  keys = [key for key in sorted_keys]
  positions = np.arange(len(combined_scores.items()))
  width = 0.3

  fig = plt.figure(figsize=(12, 6))
  ax1 = fig.add_subplot(1,1,1)

  plot_llm_scores = normalize_scores({key: np.exp(llm_scores[key]) for key in sorted_keys})
  plot_llm_scores = np.asarray([plot_llm_scores[key] for key in sorted_keys])
  plot_affordance_scores = np.asarray([vfs[key] for key in sorted_keys])
  plot_combined_scores = np.asarray([combined_scores[key] for key in sorted_keys])
  
  ax1.bar(positions, plot_combined_scores, 3 * width, alpha=0.6, color="#93CE8E", label="combined")
    
  score_colors = ["#ea9999ff" for score in plot_affordance_scores]
  ax1.bar(positions + width / 2, 0 * plot_combined_scores, width, color="#ea9999ff", label="vfs")
  ax1.bar(positions + width / 2, 0 * plot_combined_scores, width, color="#a4c2f4ff", label="language")
  ax1.bar(positions - width / 2, np.abs(plot_affordance_scores), width, color=score_colors)
  
  plt.xticks(rotation="vertical")
  ax1.set_ylim(0.0, 1.0)

  ax1.grid(True, which="both")
  ax1.axis("on")

  ax1_llm = ax1.twinx()
  ax1_llm.bar(positions + width / 2, plot_llm_scores, width, color="#a4c2f4ff", label="language")
  ax1_llm.set_ylim(0.01, 1.0)
  plt.yscale("log")
  
  font = {"fontname":"Arial", "size":"16", "color":"k" if correct else "r"}
  plt.title(task, **font)
  key_strings = [key.replace("robot.pick_and_place", "").replace(", ", " to ").replace("(", "").replace(")","") for key in keys]
  plt.xticks(positions, key_strings, **font)
  ax1.legend()
  plt.show()

#@title Affordance Scoring
#@markdown Given this environment does not have RL-trained policies or an asscociated value function, we use affordances through an object detector.

def affordance_scoring(options, found_objects, verbose=False, block_name="box", bowl_name="circle", termination_string="done()"):
  affordance_scores = {}
  found_objects = [
                   found_object.replace(block_name, "block").replace(bowl_name, "bowl") 
                   for found_object in found_objects + list(PLACE_TARGETS.keys())[-5:]]
  verbose and print("found_objects", found_objects)
  for option in options:
    if option == termination_string:
      affordance_scores[option] = 0.2
      continue
    pick, place = option.replace("robot.pick_and_place(", "").replace(")", "").split(", ")
    affordance = 0
    found_objects_copy = found_objects.copy()
    if pick in found_objects_copy:
      found_objects_copy.remove(pick)
      if place in found_objects_copy:
        affordance = 1
    affordance_scores[option] = affordance
    verbose and print(affordance, '\t', option)
  return affordance_scores

#@title Test
termination_string = "done()"
query = "To pick the blue block and put it on the red block, I should:\n"

options = make_options(PICK_TARGETS, PLACE_TARGETS, termination_string=termination_string)
llm_scores, _ = gpt3_scoring(query, options, verbose=True, engine=ENGINE)

affordance_scores = affordance_scoring(options, found_objects, block_name="box", bowl_name="circle", verbose=False, termination_string=termination_string)

combined_scores = {option: np.exp(llm_scores[option]) * affordance_scores[option] for option in options}
combined_scores = normalize_scores(combined_scores)
selected_task = max(combined_scores, key=combined_scores.get)
print("Selecting: ", selected_task)