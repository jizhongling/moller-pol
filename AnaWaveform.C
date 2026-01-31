void AnaWaveform(const Int_t proc = 0)
{
  gErrorIgnoreLevel = kWarning;
  Int_t runnumber = 242;
  const bool normalize = false;

  const UInt_t mode = 10;
  const UInt_t slot = 3;
  const UInt_t NUMSAMPLE = 240 / 4;
  const UInt_t threshold = 50;
  const UInt_t peak_cut = 300;

  const Int_t np = 3;
  Int_t ntp_event, ntp_sample[4][NUMSAMPLE], ntp_time[4][np], ntp_peak[4][np], ntp_fwhm[4][np], ntp_area[4][np], ntp_diff[6][np];
  Float_t ntp_gaus_mean[4][np], ntp_gaus_sigma[4][np], ntp_gaus_amplitude[4][np], ntp_gaus_diff[6][np];
  Float_t ntp_plot_mean[8][np], ntp_plot_sigma[8][np], ntp_plot_amplitude[8][np];
  Float_t ntp_area_sum;
  auto f_out = new TFile(Form("data/training-%d.root", proc), "RECREATE");
  auto t_out = new TTree("T", "Waveform data");
  t_out->Branch("event", &ntp_event, "event/I");
  for (Int_t ich = 0; ich < 4; ich++)
  {
    t_out->Branch(Form("sample_ch%d", ich), (Int_t *)ntp_sample[ich], Form("sample_ch%d[%d]/I", ich, NUMSAMPLE));
    for (Int_t ip = 0; ip < np; ip++)
    {
      t_out->Branch(Form("time_ch%d_p%d", ich, ip), &ntp_time[ich][ip], Form("time_ch%d_p%d/I", ich, ip));
      t_out->Branch(Form("peak_ch%d_p%d", ich, ip), &ntp_peak[ich][ip], Form("peak_ch%d_p%d/I", ich, ip));
      t_out->Branch(Form("fwhm_ch%d_p%d", ich, ip), &ntp_fwhm[ich][ip], Form("fwhm_ch%d_p%d/I", ich, ip));
      t_out->Branch(Form("area_ch%d_p%d", ich, ip), &ntp_area[ich][ip], Form("area_ch%d_p%d/I", ich, ip));
      t_out->Branch(Form("gaus_mean_ch%d_p%d", ich, ip), &ntp_gaus_mean[ich][ip], Form("gaus_mean_ch%d_p%d/F", ich, ip));
      t_out->Branch(Form("gaus_sigma_ch%d_p%d", ich, ip), &ntp_gaus_sigma[ich][ip], Form("gaus_sigma_ch%d_p%d/F", ich, ip));
      t_out->Branch(Form("gaus_amplitude_ch%d_p%d", ich, ip), &ntp_gaus_amplitude[ich][ip], Form("gaus_amplitude_ch%d_p%d/F", ich, ip));
    }
  }
  for (Int_t ich = 0; ich < 8; ich++)
    for (Int_t ip = 0; ip < np; ip++)
    {
      t_out->Branch(Form("plot_mean_ch%d_p%d", ich, ip), &ntp_plot_mean[ich][ip], Form("plot_mean_ch%d_p%d/F", ich, ip));
      t_out->Branch(Form("plot_sigma_ch%d_p%d", ich, ip), &ntp_plot_sigma[ich][ip], Form("plot_sigma_ch%d_p%d/F", ich, ip));
      t_out->Branch(Form("plot_amplitude_ch%d_p%d", ich, ip), &ntp_plot_amplitude[ich][ip], Form("plot_amplitude_ch%d_p%d/F", ich, ip));
    }
  for (Int_t ich = 0; ich < 6; ich++)
    for (Int_t ip = 0; ip < np; ip++)
    {
      t_out->Branch(Form("diff_ch%d_p%d", ich, ip), &ntp_diff[ich][ip], Form("diff_ch%d_p%d/I", ich, ip));
      t_out->Branch(Form("gaus_diff_ch%d_p%d", ich, ip), &ntp_gaus_diff[ich][ip], Form("gaus_diff_ch%d_p%d/F", ich, ip));
    }
  t_out->Branch("area_sum", &ntp_area_sum, "area_sum/F");

  auto f = new TFile(Form("Rootfiles/fadc_data_%d.root", runnumber));
  TDirectory *dir = (TDirectory *)f->Get(Form("/mode_%u_data/slot_%u", mode, slot));
  TTree *t_store = (TTree *)dir->Get("waveform");
  UInt_t store_event, store_channel, store_sample[100];
  t_store->SetBranchAddress("event", &store_event);
  t_store->SetBranchAddress("channel", &store_channel);
  t_store->SetBranchAddress("sample", store_sample);

  UInt_t last_event = 0;
  UInt_t total_channel = 0;
  UInt_t fadc_channel = 0;
  set<UInt_t> processed_channels;
  // per-event peak storage: for each of 8 channels store local-peak indices and values
  vector<vector<Int_t>> sample(8);
  vector<vector<Int_t>> time(8);
  vector<vector<Int_t>> peak(8);
  vector<vector<Int_t>> fwhm(8);
  vector<vector<Int_t>> area(8);
  UInt_t max_index[8] = {};
  UInt_t max_sample[8] = {};
  Float_t sum_sample[8] = {};
  bool trig[2] = {};

  for (Int_t ic = 0; ic < 4; ic++)
  {
    for (Int_t is = 0; is < NUMSAMPLE; is++)
      ntp_sample[ic][is] = 0;
    for (Int_t ip = 0; ip < np; ip++)
    {
      ntp_time[ic][ip] = 0;
      ntp_peak[ic][ip] = 0;
      ntp_fwhm[ic][ip] = 0;
      ntp_area[ic][ip] = 0;
      ntp_gaus_mean[ic][ip] = 0;
      ntp_gaus_sigma[ic][ip] = 0;
      ntp_gaus_amplitude[ic][ip] = 0;
    }
  }
  for (Int_t ic = 0; ic < 8; ic++)
    for (Int_t ip = 0; ip < np; ip++)
    {
      ntp_plot_mean[ic][ip] = 0;
      ntp_plot_sigma[ic][ip] = 0;
      ntp_plot_amplitude[ic][ip] = 0;
    }
  for (Int_t ic = 0; ic < 6; ic++)
    for (Int_t ip = 0; ip < np; ip++)
    {
      ntp_diff[ic][ip] = 0;
      ntp_gaus_diff[ic][ip] = 0;
    }
  ntp_area_sum = 0;

  for (ULong64_t ien = 0; ien < t_store->GetEntries(); ien++)
  {
    t_store->GetEntry(ien);
    if (ien == 0)
      last_event = store_event;

    if (store_event != last_event)
    {
      if (trig[0] && trig[1])
      {
        for (const auto &chan : processed_channels)
        {
          if (time[chan].size() > 0)
          {
            UInt_t ntp_chan = 0;
            if (sum_sample[0] + sum_sample[5] > sum_sample[1] + sum_sample[4])
            {
              switch (chan)
              {
              case 0:
                ntp_chan = sum_sample[0] > sum_sample[5] ? 0 : 1;
                break;
              case 5:
                ntp_chan = sum_sample[0] > sum_sample[5] ? 1 : 0;
                break;
              case 1:
                ntp_chan = sum_sample[1] > sum_sample[4] ? 2 : 3;
                break;
              case 4:
                ntp_chan = sum_sample[1] > sum_sample[4] ? 3 : 2;
                break;
              }
            }
            else
            {
              switch (chan)
              {
              case 0:
                ntp_chan = sum_sample[0] > sum_sample[5] ? 2 : 3;
                break;
              case 5:
                ntp_chan = sum_sample[0] > sum_sample[5] ? 3 : 2;
                break;
              case 1:
                ntp_chan = sum_sample[1] > sum_sample[4] ? 0 : 1;
                break;
              case 4:
                ntp_chan = sum_sample[1] > sum_sample[4] ? 1 : 0;
                break;
              }
            }

            auto &va = area[chan];
            vector<size_t> idx(va.size());
            iota(idx.begin(), idx.end(), 0);
            sort(idx.begin(), idx.end(),
                 [&va](size_t i1, size_t i2)
                 {
                   return va.at(i1) > va.at(i2);
                 });

            ntp_event = last_event;
            for (Int_t is = 0; is < NUMSAMPLE; is++)
              ntp_sample[ntp_chan][is] = sample[chan].at(is);

            // Fit Gaussian around peaks
            for (size_t ip = 0; ip < idx.size() && ip < np; ip++)
            {
              UInt_t ip_sorted = idx.at(ip);
              ntp_time[ntp_chan][ip] = max(time[chan].at(ip_sorted), 0);
              ntp_peak[ntp_chan][ip] = max(peak[chan].at(ip_sorted), 0);
              ntp_fwhm[ntp_chan][ip] = max(fwhm[chan].at(ip_sorted), 0);
              ntp_area[ntp_chan][ip] = max(area[chan].at(ip_sorted), 0);
              ntp_area_sum += ntp_area[ntp_chan][ip];

              // Gaussian fit around peak
              Int_t peak_time = time[chan].at(ip_sorted);
              if (peak_time > 0 && peak_time < NUMSAMPLE)
              {
                // Define fit range around peak (e.g., +/- 10 samples or based on FWHM)
                Int_t fit_range = min(10, fwhm[chan].at(ip_sorted) * 2);
                Int_t fit_start = max(0, peak_time - fit_range);
                Int_t fit_end = min((Int_t)NUMSAMPLE - 1, peak_time + fit_range);

                // Create a temporary histogram for fitting
                TH1D *h_peak = new TH1D(Form("h_peak_evt%d_ch%d_p%zu", last_event, ntp_chan, ip),
                                        "Peak", fit_end - fit_start + 1, fit_start - 0.5, fit_end + 0.5);
                for (Int_t is = fit_start; is <= fit_end; is++)
                {
                  h_peak->SetBinContent(is - fit_start + 1, sample[chan].at(is));
                }

                // Create and fit Gaussian
                TF1 *gaus_fit = new TF1(Form("gaus_evt%d_ch%d_p%zu", last_event, ntp_chan, ip),
                                        "gaus", fit_start, fit_end);
                gaus_fit->SetParameters(peak[chan].at(ip_sorted), peak_time, fwhm[chan].at(ip_sorted));
                h_peak->Fit(gaus_fit, "QN"); // Q = quiet, N = no draw

                ntp_gaus_amplitude[ntp_chan][ip] = ntp_plot_amplitude[chan][ip] = gaus_fit->GetParameter(0);
                ntp_gaus_mean[ntp_chan][ip] = ntp_plot_mean[chan][ip] = gaus_fit->GetParameter(1);
                ntp_gaus_sigma[ntp_chan][ip] = ntp_plot_sigma[chan][ip] = gaus_fit->GetParameter(2);

                delete gaus_fit;
                delete h_peak;
              }
              else
              {
                ntp_gaus_amplitude[ntp_chan][ip] = ntp_plot_amplitude[chan][ip] = 0;
                ntp_gaus_mean[ntp_chan][ip] = ntp_plot_mean[chan][ip] = 0;
                ntp_gaus_sigma[ntp_chan][ip] = ntp_plot_sigma[chan][ip] = 0;
              }
            }
          } // time[chan].size() > 0
        } // chan loop

        for (UInt_t ip = 0; ip < np; ip++)
        {
          if (ntp_time[0][ip] > 0 && ntp_time[1][ip] > 0)
            ntp_diff[0][ip] = TMath::Abs(ntp_time[0][ip] - ntp_time[1][ip]);
          if (ntp_time[2][ip] > 0 && ntp_time[3][ip] > 0)
            ntp_diff[1][ip] = TMath::Abs(ntp_time[2][ip] - ntp_time[3][ip]);
          if (ntp_time[0][ip] > 0 && ntp_time[2][ip] > 0)
            ntp_diff[2][ip] = TMath::Abs(ntp_time[0][ip] - ntp_time[2][ip]);
          if (ntp_time[0][ip] > 0 && ntp_time[2][ip] > 0)
            ntp_diff[2][ip] = TMath::Abs(ntp_time[0][ip] - ntp_time[2][ip]);
          if (ntp_time[1][ip] > 0 && ntp_time[3][ip] > 0)
            ntp_diff[3][ip] = TMath::Abs(ntp_time[1][ip] - ntp_time[3][ip]);
          if (ntp_time[0][ip] > 0 && ntp_time[3][ip] > 0)
            ntp_diff[4][ip] = TMath::Abs(ntp_time[0][ip] - ntp_time[3][ip]);
          if (ntp_time[1][ip] > 0 && ntp_time[2][ip] > 0)
            ntp_diff[5][ip] = TMath::Abs(ntp_time[1][ip] - ntp_time[2][ip]);

          if (ntp_gaus_mean[0][ip] > 0 && ntp_gaus_mean[1][ip] > 0)
            ntp_gaus_diff[0][ip] = TMath::Abs(ntp_gaus_mean[0][ip] - ntp_gaus_mean[1][ip]);
          if (ntp_gaus_mean[2][ip] > 0 && ntp_gaus_mean[3][ip] > 0)
            ntp_gaus_diff[1][ip] = TMath::Abs(ntp_gaus_mean[2][ip] - ntp_gaus_mean[3][ip]);
          if (ntp_gaus_mean[0][ip] > 0 && ntp_gaus_mean[2][ip] > 0)
            ntp_gaus_diff[2][ip] = TMath::Abs(ntp_gaus_mean[0][ip] - ntp_gaus_mean[2][ip]);
          if (ntp_gaus_mean[1][ip] > 0 && ntp_gaus_mean[3][ip] > 0)
            ntp_gaus_diff[3][ip] = TMath::Abs(ntp_gaus_mean[1][ip] - ntp_gaus_mean[3][ip]);
          if (ntp_gaus_mean[0][ip] > 0 && ntp_gaus_mean[3][ip] > 0)
            ntp_gaus_diff[4][ip] = TMath::Abs(ntp_gaus_mean[0][ip] - ntp_gaus_mean[3][ip]);
          if (ntp_gaus_mean[1][ip] > 0 && ntp_gaus_mean[2][ip] > 0)
            ntp_gaus_diff[5][ip] = TMath::Abs(ntp_gaus_mean[1][ip] - ntp_gaus_mean[2][ip]);
        }

        if (normalize)
          for (Int_t ich = 0; ich < 4; ich++)
            for (size_t ip = 0; ip < np; ip++)
            {
              ntp_time[ich][ip] /= 10;
              ntp_peak[ich][ip] /= 500;
              ntp_fwhm[ich][ip] /= 1;
              ntp_area[ich][ip] /= 1000;
            }

        t_out->Fill();
      } // trig[0] && trig[1]

      total_channel = -1;
      fadc_channel = 0;
      processed_channels.clear();
      for (Int_t ic = 0; ic < 8; ic++)
      {
        max_index[ic] = 0;
        max_sample[ic] = 0;
        sum_sample[ic] = 0;
        // clear any stored peaks for this channel for the next event
        sample[ic].clear();
        time[ic].clear();
        peak[ic].clear();
        fwhm[ic].clear();
        area[ic].clear();
      }
      for (Int_t ic = 0; ic < 4; ic++)
      {
        for (Int_t is = 0; is < NUMSAMPLE; is++)
          ntp_sample[ic][is] = 0;
        for (Int_t ip = 0; ip < np; ip++)
        {
          ntp_time[ic][ip] = 0;
          ntp_peak[ic][ip] = 0;
          ntp_fwhm[ic][ip] = 0;
          ntp_area[ic][ip] = 0;
          ntp_gaus_mean[ic][ip] = 0;
          ntp_gaus_sigma[ic][ip] = 0;
          ntp_gaus_amplitude[ic][ip] = 0;
        }
      }
      for (Int_t ic = 0; ic < 8; ic++)
        for (Int_t ip = 0; ip < np; ip++)
        {
          ntp_plot_mean[ic][ip] = 0;
          ntp_plot_sigma[ic][ip] = 0;
          ntp_plot_amplitude[ic][ip] = 0;
        }
      for (Int_t ic = 0; ic < 6; ic++)
        for (Int_t ip = 0; ip < np; ip++)
        {
          ntp_diff[ic][ip] = 0;
          ntp_gaus_diff[ic][ip] = 0;
        }
      ntp_area_sum = 0;
      for (Int_t it = 0; it < 2; it++)
        trig[it] = 0;

      last_event = store_event;
    } // new event

    if (store_channel < 8 && processed_channels.find(store_channel) == processed_channels.end())
    {
      processed_channels.insert(store_channel);

      const Int_t nped = 4;
      Float_t lped = 0., rped = 0.;
      for (Int_t is = 0; is < nped; is++)
        lped += store_sample[is];
      for (Int_t is = NUMSAMPLE - nped; is < NUMSAMPLE; is++)
        rped += store_sample[is];
      Float_t ped = min(lped, rped) / (Float_t)nped;

      for (Int_t is = 0; is < NUMSAMPLE; is++)
      {
        sample[store_channel].push_back(store_sample[is] - ped);
        if (store_sample[is] - ped > max_sample[store_channel])
        {
          max_index[store_channel] = is;
          max_sample[store_channel] = store_sample[is] - ped;
        }
        sum_sample[store_channel] += store_sample[is] - ped;
      }

      if (max_sample[store_channel] > peak_cut)
      {
        fadc_channel++;
        trig[store_channel / 4] = true;
      }

      // detect local maxima (handles flat-top peaks)
      for (Int_t is = 1; is < (Int_t)NUMSAMPLE - 1; is++)
      {
        if (store_sample[is] - ped > peak_cut)
        {
          // Check if we're at a local maximum (including flat tops)
          // Find the extent of any plateau at this level
          Int_t plateau_start = is;
          Int_t plateau_end = is;

          // Extend plateau backwards while values are equal
          while (plateau_start > 0 && store_sample[plateau_start - 1] == store_sample[is])
            plateau_start--;

          // Extend plateau forwards while values are equal
          while (plateau_end < (Int_t)NUMSAMPLE - 1 && store_sample[plateau_end + 1] == store_sample[is])
            plateau_end++;

          // Check if this plateau is a local maximum
          bool is_peak = true;
          if (plateau_start > 0 && store_sample[plateau_start - 1] >= store_sample[is])
            is_peak = false;
          if (plateau_end < (Int_t)NUMSAMPLE - 1 && store_sample[plateau_end + 1] >= store_sample[is])
            is_peak = false;

          if (is_peak && is == plateau_start) // Only process once per plateau
          {
            // Use the center of the plateau as peak position
            Int_t peak_pos = (plateau_start + plateau_end) / 2;

            time[store_channel].push_back(peak_pos);
            peak[store_channel].push_back(store_sample[is] - ped);

            Int_t ld = 0, rd = 0;
            Float_t sum_area = 0.;
            for (Int_t id = 0; id < NUMSAMPLE; id++)
              if (peak_pos - id >= 0 && peak_pos - id < (Int_t)NUMSAMPLE)
              {
                if (store_sample[peak_pos - id] - ped > (store_sample[is] - ped) / 2)
                  ld++;
                else
                  break;
              }
            for (Int_t id = 0; id < NUMSAMPLE; id++)
              if (peak_pos + id >= 0 && peak_pos + id < (Int_t)NUMSAMPLE)
              {
                if (store_sample[peak_pos + id] - ped > (store_sample[is] - ped) / 2)
                  rd++;
                else
                  break;
              }
            fwhm[store_channel].push_back(rd + ld);
            for (Int_t id = -2 * ld; id <= 2 * rd; id++)
              if (peak_pos + id >= 0 && peak_pos + id < (Int_t)NUMSAMPLE)
                sum_area += store_sample[peak_pos + id] - ped;
            area[store_channel].push_back(sum_area);
          }

          // Skip to end of plateau to avoid reprocessing
          is = plateau_end;
        }
      } // is
    } // PMT channels

    total_channel++;
  } // ien

  f_out->Write();
  f_out->Close();
}
