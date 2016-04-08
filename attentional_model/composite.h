class CompositeModel : public cnn::Model
{
public:
    CompositeModel(const std::vector<Model*> &models)
	: cnn::Model(), m_models(models) 
    {
	for (auto m: models) {
	    std::copy(m.parameters_list().begin(), m.parameters_list().end(),
		    std::back_inserter(m_params, m_params.begin()));
	    std::copy(m.lookup_parameters_list().begin(), m.lookup_parameters_list().end(),
		    std::back_inserter(m_lookup_params, m_lookup_params.begin()));
	}

	m_all_params = m_params;
	std::copy(m_lookup_parameters.begin(), m.lookup_parameters.end(),
		std::back_inserter(m_all_params, m_all_params.begin()));
    }

    ~CompositeModel() 
    {
	for (auto m: models)
	    delete m;
    }

    const std::vector<ParametersBase*>& all_parameters_list() const { return m_all_params; }
    const std::vector<Parameters*>& parameters_list() const { return m_params; }
    const std::vector<LookupParameters*>& lookup_parameters_list() const { return m_lookup_params; }

private:
    std::vector<Model*> m_models;
    std::vector<ParametersBase*> m_all_params;
    std::vector<Parameters*> m_params;
    std::vector<LookupParameters*> m_lookup_params;
};

