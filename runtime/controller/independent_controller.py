from tqdm import tqdm

from runtime.agent.agent import GlobalStep
from runtime.controller.controller import AController


class IndependentController(AController):

    def search(self, number_of_episodes, model_reference):
        initial_model_for_metrics = self._model_factory.to_executable_model(model_reference)
        reference_evaluation = self._model_evaluator.evaluate(initial_model_for_metrics, list()) | self._lat_eval(
            initial_model_for_metrics)
        self._agent_wrapper.initialize(reference_evaluation)
        self._logging_service.search_started(number_of_episodes, reference_evaluation)

        # we compress here using the reference policy to support passing in existing policies
        # for normal (initial) reference policies this will have no compression effect
        _, initial_model_for_features = self._compress_adapter.compress(self._reference_policy, model_reference)
        reference_features = self._feature_extractor.extract_metrics_and_sens(initial_model_for_features,
                                                                              model_reference)

        for episode in tqdm(range(number_of_episodes), desc="[Search Policies - Episodes]"):
            # difference
            episode_features = reference_features
            episode_policy = self._reference_policy

            self._logging_service.episode_started(episode)
            for step in tqdm(range(self._steps_per_episode), desc="[Episode Steps]"):
                global_step = GlobalStep(step=step, episode=episode)
                predicted_policy, raw_actions = self._agent_wrapper.predict_policy(global_step, episode_features,
                                                                                   episode_policy)

                if predicted_policy is not None:
                    episode_policy = predicted_policy
                    executable_model, step_evaluation = self._compress_and_evaluate(episode_policy, model_reference)
                    # difference
                    episode_features = self._feature_extractor.update_metrics(executable_model, episode_features)

                    self._agent_wrapper.pass_step_results(global_step, step_evaluation, episode_features)
                    self._logging_service.step_results(step_evaluation, raw_actions)

            episode_model, episode_evaluation = self._compress_and_evaluate(episode_policy, model_reference,
                                                                            episode=True)
            optimization_result = self._agent_wrapper.pass_episode_results(episode, episode_evaluation,
                                                                           episode_features)
            self._logging_service.episode_completed(episode_model.applied_policy, episode_evaluation,
                                                    optimization_result, episode_model.compression_protocol)
