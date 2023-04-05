from jax.scipy.linalg import svd

def compute_eval_metrics(params, batch, n_samples):
# compute the time for inference.
  duration_list = []
  accuracy_list = []
  for _ in range(n_samples):
    #start = 0.0
    # compute the accuracy on a given batch.
    start_time = time.time()
    acc = compute_accuracy(params, batch)
    duration = time.time() - start_time
    duration_list.append(duration)
    accuracy_list.append(acc)

  return accuracy_list, duration_list

 def rank_approximated_weight(weight: jnp.ndarray, rank_fraction: float):
  # compute the SVD of the matrix to return the rank approximated weights u and v for a given matrix.
  
  size = weight.shape[1]

  # SVD
  U, s, VT = svd(weight, full_matrices=True)

  # k rank
  n_elements = int(jnp.floor((rank_fraction*weight.shape[1]*weight.shape[0]) / (weight.shape[0]+weight.shape[1])))

  # new sigma matrix by removing lower sigma values
  diag_s = jnp.diag(s)
  Sigma = jnp.zeros((weight.shape[0], weight.shape[1]))
  for i in np.arange(n_elements):
    Sigma = Sigma.at[i,i].set(diag_s[i,i]) # update Sigma matrix
  
  return U.dot(Sigma), VT


rank_truncated_params = deepcopy(params)
ranks_and_accuracies = []
ranks_and_times = []
for rank_fraction in np.arange(1.0, 0.0, -0.1):

  print(f"Evaluating the model at {rank_fraction}")
  for layer in params.keys():
    if 'conv' in layer:
      continue
    weight = params[layer]['w']
    # rank_approximated_weight function to compute the SVD of the matrix to return the rank approximated weights u and v for a given matrix.
    u, v = rank_approximated_weight(weight, rank_fraction)
    rank_truncated_params[layer]['w'] = u@v

  test_batch = next(test)
  # we compute metrics over 50 samples to reduce noise in the measurement.
  n_samples = 50
  # compute_eval_metrics function to compute latency 50 seperate times given the batch passed to compute_eval_metrics. Return the average across all latencies you store.
  test_accuracy, latency = compute_eval_metrics(rank_truncated_params, next(test), n_samples)
  print(f"Rank Fraction / Test accuracy: "
          f"{rank_fraction:.2f} / {np.mean(test_accuracy):.3f}.")
  ranks_and_accuracies.append((rank_fraction, np.mean(test_accuracy)))
  print(f"Rank Fraction / Duration: "
          f"{rank_fraction:.2f} / {np.mean(latency):.4f}.")
  ranks_and_times.append((rank_fraction, np.mean(latency)))


# Plot relationship between rank fraction and test accuracy
plt.plot(list(zip(*ranks_and_accuracies))[0], list(zip(*ranks_and_accuracies))[1])
plt.ylabel('Accuracy')
plt.xlabel('Rank Fraction')
plt.show()

# Does replacing the weight matrix with the low factor matrix result in latency speed ups?
plt.plot(list(zip(*ranks_and_times))[0], list(zip(*ranks_and_times))[1])
plt.ylabel('Time')
plt.xlabel('Rank %')
plt.show()



