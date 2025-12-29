void AnaWaveform(const Int_t proc = 0)
{
  gErrorIgnoreLevel = kWarning;
  Int_t runnumber = 242;
  const bool normalize = false;

  const Int_t mode = 10;
  const UInt_t slot = 3;
  const UInt_t NUMSAMPLE = 240 / 4;
  const UInt_t threshold = 50;
  const UInt_t peak_cut = 150;

  const Int_t np = 3;
  Int_t ntp_event, ntp_time[4][np], ntp_peak[4][np], ntp_fwzm[4][np], ntp_area[4][np], ntp_diff[6][np];
  auto f_out = new TFile(Form("data/training-%d.root", proc), "RECREATE");
  auto t_out = new TTree("T", "Waveform data");
  t_out->Branch("event", &ntp_event, "event/I");
  for (Int_t ip = 0; ip < np; ip++)
  {
    for (Int_t ich = 0; ich < 4; ich++)
    {
      t_out->Branch(Form("time_ch%d_p%d", ich, ip), &ntp_time[ich][ip], Form("time_ch%d_p%d/I", ich, ip));
      t_out->Branch(Form("peak_ch%d_p%d", ich, ip), &ntp_peak[ich][ip], Form("peak_ch%d_p%d/I", ich, ip));
      t_out->Branch(Form("fwzm_ch%d_p%d", ich, ip), &ntp_fwzm[ich][ip], Form("fwzm_ch%d_p%d/I", ich, ip));
      t_out->Branch(Form("area_ch%d_p%d", ich, ip), &ntp_area[ich][ip], Form("area_ch%d_p%d/I", ich, ip));
    }
    for (Int_t ich = 0; ich < 6; ich++)
    {
      t_out->Branch(Form("diff_ch%d_p%d", ich, ip), &ntp_diff[ich][ip], Form("diff_ch%d_p%d/I", ich, ip));
    }
  }

  auto f = new TFile(Form("Rootfiles/fadc_data_%d.root", runnumber));
  TDirectory *dir = (TDirectory *)f->Get(Form("/mode_%d_data/slot_%u", mode, slot));
  TTree *t_store = (TTree *)dir->Get("waveform");
  UInt_t store_event, store_channel, store_sample[100];
  t_store->SetBranchAddress("event", &store_event);
  t_store->SetBranchAddress("channel", &store_channel);
  t_store->SetBranchAddress("sample", store_sample);

  UInt_t last_event = 0;
  UInt_t total_channel = 0;
  UInt_t fadc_channel = 0;
  vector<UInt_t> v_channel;
  // per-event peak storage: for each of 8 channels store local-peak indices and values
  vector<vector<Int_t>> time(8);
  vector<vector<Int_t>> peak(8);
  vector<vector<Int_t>> fwzm(8);
  vector<vector<Int_t>> area(8);
  UInt_t max_index[8] = {};
  UInt_t max_sample[8] = {};
  Float_t sum_sample[8] = {};
  bool trig[2] = {};

  for (Int_t ip = 0; ip < np; ip++)
  {
    for (Int_t ic = 0; ic < 4; ic++)
    {
      ntp_time[ic][ip] = 0;
      ntp_peak[ic][ip] = 0;
      ntp_fwzm[ic][ip] = 0;
      ntp_area[ic][ip] = 0;
    }
    for (Int_t ic = 0; ic < 6; ic++)
    {
      ntp_diff[ic][ip] = 0;
    }
  }

  for (ULong64_t ien = 0; ien < t_store->GetEntries(); ien++)
  {
    t_store->GetEntry(ien);

    if (store_event != last_event)
    {
      last_event = store_event;
      ien--;

      if (trig[0] && trig[1])
      {
        for (ULong64_t jen = ien + 1 - total_channel; jen < ien + 1; jen++)
        {
          t_store->GetEntry(jen);
          // cout << store_event << ", " << store_channel << ", " << store_sample[0] << endl;
          if (store_channel < 8)
          {
            v_channel.push_back(store_channel);
          } // PMT channels
        } // jen

        for (UInt_t ich = 0; ich < v_channel.size(); ich++)
        {
          UInt_t chan = v_channel.at(ich);
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

            ntp_event = store_event;
            for (size_t ip = 0; ip < time[chan].size() && ip < np; ip++)
            {
              ntp_time[ntp_chan][ip] = time[chan].at(ip);
              ntp_peak[ntp_chan][ip] = peak[chan].at(ip);
              ntp_fwzm[ntp_chan][ip] = fwzm[chan].at(ip);
              ntp_area[ntp_chan][ip] = area[chan].at(ip);
            }
          } // time[chan].size() > 0
        } // ich

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
        }

        if (normalize)
          for (Int_t ich = 0; ich < 4; ich++)
            for (size_t ip = 0; ip < np; ip++)
            {
              ntp_time[ich][ip] /= 10;
              ntp_peak[ich][ip] /= 500;
              ntp_fwzm[ich][ip] /= 3;
              ntp_area[ich][ip] /= 1000;
            }

        t_out->Fill();
        v_channel.clear();
      } // trig[0] && trig[1]

      total_channel = -1;
      fadc_channel = 0;
      for (Int_t ic = 0; ic < 8; ic++)
      {
        max_index[ic] = 0;
        max_sample[ic] = 0;
        sum_sample[ic] = 0;
        // clear any stored peaks for this channel for the next event
        time[ic].clear();
        peak[ic].clear();
        fwzm[ic].clear();
        area[ic].clear();
      }
      for (Int_t ip = 0; ip < np; ip++)
      {
        for (Int_t ic = 0; ic < 4; ic++)
        {
          ntp_time[ic][ip] = 0;
          ntp_peak[ic][ip] = 0;
          ntp_fwzm[ic][ip] = 0;
          ntp_area[ic][ip] = 0;
        }
        for (Int_t ic = 0; ic < 6; ic++)
        {
          ntp_diff[ic][ip] = 0;
        }
      }
      for (Int_t it = 0; it < 2; it++)
        trig[it] = 0;
    } // new event

    else if (store_channel < 8)
    {
      const Int_t nped = 4;
      Float_t ped = 0.;
      for (Int_t is = 0; is < nped; is++)
        ped += store_sample[is];
      ped /= (Float_t)nped;

      for (Int_t is = 0; is < NUMSAMPLE; is++)
      {
        // detect local maxima (simple 1-sample neighbor check)
        if (is > 0 && is < (Int_t)NUMSAMPLE - 1)
        {
          if (store_sample[is] - ped > peak_cut &&
              store_sample[is] > store_sample[is - 1] &&
              store_sample[is] > store_sample[is + 1])
          {
            time[store_channel].push_back(is);
            peak[store_channel].push_back(store_sample[is] - ped);
            Int_t ld = 0, rd = 0;
            Float_t sum_area = 0.;
            for (Int_t id = 0; id < NUMSAMPLE; id++)
              if (is - id >= 0 && is - id < (Int_t)NUMSAMPLE)
              {
                if (store_sample[is - id] - ped > threshold)
                  ld++;
                else
                  break;
              }
            for (Int_t id = 0; id < NUMSAMPLE; id++)
              if (is + id >= 0 && is + id < (Int_t)NUMSAMPLE)
              {
                if (store_sample[is + id] - ped > threshold)
                  rd++;
                else
                  break;
              }
            fwzm[store_channel].push_back(rd - ld);
            for (Int_t id = -ld + 1; id < rd; id++)
              sum_area += store_sample[is + id] - ped;
            area[store_channel].push_back(sum_area);
          }
        }

        if (store_sample[is] - ped > max_sample[store_channel])
        {
          max_index[store_channel] = is;
          max_sample[store_channel] = store_sample[is];
        }
        sum_sample[store_channel] += store_sample[is] - ped;
      } // is

      if (max_sample[store_channel] > peak_cut)
      {
        fadc_channel++;
        trig[store_channel / 4] = true;
      }
    } // PMT channels

    total_channel++;
  } // ien

  f_out->Write();
  f_out->Close();
}
